# -*- coding: utf-8 -*-
"""The model configuration of Iterative Neural Network (INN).

This module implements the INN model proposed by Cao (2019), Nested Relation
Extraction with Iterative Neural Network.

Example:

        $ python model.py

Authors:
    Sammy Jia - https://github.com/sajia28
    Fangzhou Li - https://github.com/fangzhouli
    Gabriel Simmons - https://github.com/g-simmons

TODO:
    * Loss
    * Batched training
    *

"""

# this file is dedicated toward holding the actual model
from os import path
import json
import pickle
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from itertools import combinations
from collections import Counter

from config import (
    ENTITY_PREFIX,
    PREDICATE_PREFIX,
    EPOCHS,
    WORD_EMBEDDING_DIM,
    VECTOR_DIM,
    HIDDEN_DIM,
    RELATION_EMBEDDING_DIM,
)

# NOTES FROM PAPER
# The final fully connected layer is 512 × 1024 × 2.
# We use Adadelta [31] as the optimizer with learning rate = 1.0.
# The batch size is 8.


class DAGLSTMCell(nn.Module):
    # credit to https://github.com/dmlc/dgl/tree/master/examples/pytorch/tree_lstm for the
    # original tree-lstm implementation, modified here
    def __init__(
        self,
        hidden_dim: int,
        relation_embedding_dim: int,
        max_inputs: int,
    ):
        super(DAGLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.relation_embedding_dim = relation_embedding_dim
        self.max_inputs = max_inputs

        self.W_ioc_hat = nn.Linear(relation_embedding_dim, 3 * hidden_dim)
        self.W_fs = nn.ModuleList(
            [
                nn.Linear(relation_embedding_dim, hidden_dim)
                for j in range(max_inputs)
            ]
        )

        self.U_ioc_hat = nn.Linear(
            max_inputs * hidden_dim, 3 * hidden_dim
        )  # can pad with zeros to have dynamic
        self.U_fs = nn.ModuleList(
            [
                nn.Linear(max_inputs * hidden_dim, hidden_dim)
                for j in range(max_inputs)
            ]
        )

        self.b_ioc_hat = nn.Parameter(th.zeros(3 * hidden_dim))
        self.b_fs = nn.ParameterList(
            [nn.Parameter(th.zeros(hidden_dim)) for j in range(max_inputs)]
        )

    def init_cell_state(self):
        return th.tensor(th.zeros(1, self.hidden_dim))

    def forward(self, hjs, cjs, e):
        v = th.cat(hjs)
        ioc_hat = self.W_ioc_hat(e)
        ioc_hat += self.U_ioc_hat(v)
        ioc_hat += self.b_ioc_hat
        ioc_hat =  th.sigmoid(ioc_hat)
        i, o, c_hat = th.chunk(ioc_hat, 3)
        i, o, c_hat = th.sigmoid(i), th.sigmoid(o), th.tanh(c_hat)

        fj_mul_cs = th.zeros(1, self.hidden_dim)
        for j in range(len(hjs)):
            fj = th.sigmoid(self.W_fs[j](e) + self.U_fs[j](v) + self.b_fs[j])
            fj_mul_cs += th.mul(fj, cjs[j])

        c = th.mul(i, c_hat) + th.sum(fj_mul_cs)

        h = th.mul(th.tanh(c), o)

        return h, c


class INNModel(nn.Module):
    """INN model configuration.

    Parameters:
        vocab_dict (dict): The vocabulary for training, tokens to indices.
        embedding_dim (int): The size of the embedding vector.
        hidden_dim (int): The size of LSTM hidden vector, if bi-LSTM,
            multilplied by 2

    """

    def __init__(
        self,
        vocab_dict,
        word_embedding_dim,
        relation_embedding_dim,
        hidden_dim,
        configuration,
        entity_to_idx,
        relation_to_idx,
    ):
        super().__init__()
        self.vocab_dict = vocab_dict
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.relation_embedding_dim = relation_embedding_dim
        self.configuration = configuration
        self.inverted_configuration = self._invert_configuration(
            self.configuration
        )  # probably an antipattern here :shrug:
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx
        self.elements = None
        self.candidates = []
        self.to_predict = []

        self.word_embeddings = nn.Embedding(len(vocab_dict), self.word_embedding_dim)

        self.relation_embeddings = nn.Embedding(
            len(configuration.keys()), self.relation_embedding_dim
        )

        self.attn_scores = nn.Linear(in_features=self.hidden_dim * 2, out_features=1)

        self.blstm = nn.LSTM(
            input_size=self.word_embedding_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True,
            num_layers=1,
        )

        self.cell = DAGLSTMCell(
            hidden_dim= 2 * self.hidden_dim,
            relation_embedding_dim= self.relation_embedding_dim,
            max_inputs=2
        )

        self.output_linear = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 4 * self.hidden_dim),
            nn.Linear(4 * self.hidden_dim, 2),
        )

    def get_embeddings(self, sentence):
        """
        Accepts a sentence as input and returns a list of embedding vectors, one vector per token.
        i-th embedding vector corresponds to the i-th token
        """
        token_list = sentence.split()

        index_list = []
        for token in token_list:
            if token in self.vocab_dict:
                index_list.append(self.vocab_dict[token])
            else:
                index_list.append(self.vocab_dict["UNK"])

        return self.word_embeddings(th.LongTensor(index_list))

    def get_h_entities(self, entities, blstm_out):
        """Apply attention mechanism to entity representation.

        Args:
            entities (list of tuples::(str, (int,))): A list of pairs of an
                entity label and indices of words.
            blstm_out (th.Tensor): The output hidden states of bi-LSTM.

        Returns:
            h_entities (th.Tensor): The output hidden states of entity
                representation layer.

        """
        h_entities = []
        attn_scores_out = self.attn_scores(blstm_out)
        for entity in entities:
            tok_indices = list(entity[1])
            h_entity = 0
            attn_weights = F.softmax(attn_scores_out[tok_indices])
            for i in range(len(attn_weights)):
                h_entity += attn_weights[i] * blstm_out[tok_indices][i]
            h_entities.append(h_entity)
        h_entities = th.cat(h_entities)
        return h_entities

    def _invert_configuration(self, configuration):
        inverted_configuration = {}

        for rel, argsets in configuration.items():
            for argset in argsets:
                if argset not in inverted_configuration.keys():
                    inverted_configuration[argset] = Counter()
                inverted_configuration[argset][rel] += 1

        return inverted_configuration

    def _generate_argsets(self):
        self.argsets = set(
            [(c,) for c in self.candidates.items()]
        )
        for argset_idx in combinations(self.candidates.keys(), r=2):
            if self._argset_idx_in_configuration(argset_idx):
                argset = tuple((c, self.candidates[c]) for c in argset_idx)
                self.argsets.add(argset)

    def _generate_to_predict(self):
        for argset in self.argsets:
            key = ",".join([f"{arg[0][1]}" for arg in argset])
            rels = self.inverted_configuration[key]
            for rel in rels.keys():
                self.to_predict.append(
                    (f"{PREDICATE_PREFIX}{rel}",
                    self.relation_embeddings(th.tensor(self.relation_to_idx[rel])),
                    [arg[1] for arg in argset])
                    )


    def _argset_idx_in_configuration(self, argset_idx):
        key = ",".join([f"{ENTITY_PREFIX}{ai[1]}" for ai in argset_idx])
        return key in self.inverted_configuration.keys()

    def forward(self, x):
        sentence, entities = x

        # entity_indices = [self.entity_to_idx[ent[0]] for ent in entities]
        # entity_indices = th.tensor(entity_indices)

        embedded_sentence = self.get_embeddings(sentence)

        # since bi-lstm's hidden vector is actually 2 concatenated vectors,
        #   it will be 2x as long (512)
        blstm_out, _ = self.blstm(
            embedded_sentence.view(embedded_sentence.shape[0], 1, -1)
        )

        h_entities = self.get_h_entities(entities, blstm_out)

        entity_names = [ent[0] for ent in entities]

        c = self.cell.init_cell_state()

        self.candidates = {}
        for i, e in enumerate(zip(entity_names,h_entities)):
            e_name = e[0]
            h = e[1]
            self.candidates[(i,f"{ENTITY_PREFIX}{e_name}")] = h

        new_candidates = True

        self.to_predict = []
        layer = 1

        predictions = []

        while new_candidates:
            new_candidates = False
            print(f'==Layer {layer}==')
            self.to_predict = []
            self._generate_argsets()
            self._generate_to_predict()

            for tp in self.to_predict:
                e = tp[1] # relation embedding
                hjs = tp[2] # hidden states
                cjs = [c for _ in hjs]
                if len(tp[2])>1: # TODO: deal with predicates with 1 argument
                    h, c = self.cell.forward(hjs,cjs,e)
                    logits = self.output_linear(h)
                    p = F.softmax(logits)
                    if p[0] > 0.5: #assumes 0 index is the positive class
                        self.candidates[(len(self.candidates),tp[0])] = h
                        new_candidates = True
                        predictions.append("a_prediction")
            layer += 1

        # e = self.relation_embeddings(self.relation_to_idx[relation_name])
        # h, c = self.cell.forward(hjs, cjs, e)

        positive_candidates = None
        return predictions


def get_sentences(percent_test):
    with open("text_sentences.txt") as f:
        sentences = f.read().splitlines()
    entities = pickle.load(
        open(path.abspath(path.dirname(__file__)) + "/entities.pickle", "rb")
    )

    train_data, test_data = train_test_split(
        list(zip(sentences, entities)), test_size=percent_test, random_state=0
    )

    return train_data, test_data


def main():
    train_data, test_data = get_sentences(0.2)

    vocab_dict = eval(open("vocab_dict.txt", "r").read())
    configuration = json.load(open("../data/configuration.json", "r"))
    entities_to_idx = json.load(
        open("../data/entities.json", "r")
    )  # TODO put these both in one file
    relation_to_idx = json.load(open("../data/relations.json", "r"))

    model = INNModel(
        vocab_dict,
        WORD_EMBEDDING_DIM,
        RELATION_EMBEDDING_DIM,
        HIDDEN_DIM,
        configuration,
        entities_to_idx,
        relation_to_idx,
    )

    print('\n\n',train_data[0][0],'\n\n')

    output = model.forward(train_data[0])

    print(output)

    # for epoch in range(EPOCHS):
    #     for step, x in enumerate(train_data):
    #         print("Epoch {:05d} | Step {:05d} | Loss {:.4f} |".format(
    #             epoch, step, loss.item()))

    # acc = float(th.sum(th.eq(batch.label, pred))) / len(batch.label)
    # print("Epoch {:05d} | Step {:05d} | Loss {:.4f} | Acc {:.4f} |".format(
    #     epoch, step, loss.item(), acc))

    return 0


main()
