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
    * Prediction representation & Loss
    * Add a tree structure to candidates
    * Training
    * Batched training

"""

# this file is dedicated toward holding the actual model
from os import path
import json
import pickle
import numpy as np
import hiddenlayer as hl
import torch as th
from torch import nn
from torch.nn import functional as functional
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchtext.vocab import Vocab
from sklearn.model_selection import train_test_split
from itertools import combinations
from collections import Counter, namedtuple
from tqdm import tqdm

from classes import BioInferTaskConfiguration

from config import (
    ENTITY_PREFIX,
    PREDICATE_PREFIX,
    EPOCHS,
    WORD_EMBEDDING_DIM,
    VECTOR_DIM,
    HIDDEN_DIM,
    RELATION_EMBEDDING_DIM,
    BATCH_SIZE,
)

# NOTES FROM PAPER
# The final fully connected layer is 512 × 1024 × 2.
# We use Adadelta [31] as the optimizer with learning rate = 1.0.


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

        self.W_ioc_hat = nn.Linear(relation_embedding_dim, 3 * hidden_dim, bias=False)
        self.W_fs = nn.ModuleList(
            [nn.Linear(relation_embedding_dim, hidden_dim, bias=False) for j in range(max_inputs)]
        )

        self.U_ioc_hat = nn.Linear(
            max_inputs * hidden_dim, 3 * hidden_dim, bias=False
        )  # can pad with zeros to have dynamic
        self.U_fs = nn.ModuleList(
            [nn.Linear(max_inputs * hidden_dim, hidden_dim, bias=False) for j in range(max_inputs)]
        )

        self.b_ioc_hat = nn.Parameter(th.zeros(3 * hidden_dim))
        self.b_fs = nn.ParameterList(
            [nn.Parameter(th.zeros(hidden_dim)) for j in range(max_inputs)]
        )

    def init_cell_state(self):
        return th.zeros(1, self.hidden_dim).clone().detach().requires_grad_(True)

    def forward(self, hjs, cjs, e):
        v = th.cat(hjs)
        ioc_hat = self.W_ioc_hat(e)
        ioc_hat += self.U_ioc_hat(v)
        ioc_hat += self.b_ioc_hat
        ioc_hat = th.sigmoid(ioc_hat)
        i, o, c_hat = th.chunk(ioc_hat, 3)
        i, o, c_hat = th.sigmoid(i), th.sigmoid(o), th.tanh(c_hat)

        fj_mul_cs = th.zeros(1, self.hidden_dim)
        for j in range(len(hjs)):
            fj = th.sigmoid(self.W_fs[j](e) + self.U_fs[j](v) + self.b_fs[j])
            fj_mul_cs += th.mul(fj, cjs[j])

        c = th.mul(i, c_hat) + th.sum(fj_mul_cs)

        h = th.mul(th.tanh(c), o)

        return h, c


class ElementList:
    def __init__(
        self,
    ):
        self.elements = {}

    def add_element(self, element):
        self.elements[(len(self.elements), element.name)] = element.hidden_vector

    def items(
        self,
    ):
        return self.elements.items()

    def keys(
        self,
    ):
        return self.elements.keys()

    def get_combinations(self, r):
        return combinations(self.keys(), r)

    def __getitem__(self, key):
        return self.elements[key]

    def __repr__(
        self,
    ):
        return str(self.elements)

    def __len__(
        self,
    ):
        return len(self.elements)


Element = namedtuple("Element", ["name", "hidden_vector", "argument_names"])
PredictionCandidate = namedtuple(
    "PredictionCandidate",
    ["predicate", "predicate_embedding", "argument_names", "argument_hidden_vectors"],
)


class INNModel(nn.Module):
    """INN model configuration.

    Parameters:
        vocab_dict (dict): The vocabulary for training, tokens to indices.
        word_embedding_dim (int): The size of the word embedding vectors.
        relation_embedding_dim (int): The size of the relation embedding vectors.
        hidden_dim (int): The size of LSTM hidden vector (effectively 1/2 of the desired BiLSTM output size).
        schema: The task schema
        element_to_idx (dict): dictionary mapping entity strings to unique integer values
    """

    def __init__(
        self,
        vocab_dict,
        word_embedding_dim,
        relation_embedding_dim,
        hidden_dim,
        schema,
        inverted_schema,
        element_to_idx,
        max_layer_height,
    ):
        super().__init__()
        self.vocab_dict = vocab_dict
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.relation_embedding_dim = relation_embedding_dim
        self.schema = schema
        self.inverted_schema = inverted_schema
        self.element_to_idx = element_to_idx
        self.idx_to_element = {v:k for k,v in element_to_idx.items()}
        self.max_layer_height = max_layer_height

        self.word_embeddings = nn.Embedding(len(vocab_dict), self.word_embedding_dim)

        self.element_embeddings = nn.Embedding(
            len(element_to_idx.keys()), self.relation_embedding_dim
        )

        self.attn_scores = nn.Linear(in_features=self.hidden_dim * 2, out_features=1)

        self.blstm = nn.LSTM(
            input_size=self.word_embedding_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True,
            num_layers=1,
        )

        self.cell = DAGLSTMCell(
            hidden_dim=2 * self.hidden_dim,
            relation_embedding_dim=self.relation_embedding_dim,
            max_inputs=2,
        )

        self.output_linear = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 4 * self.hidden_dim),
            nn.Linear(4 * self.hidden_dim, 2),
        )


    def get_h_entities(self, entity_indices, blstm_out):
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
        for tok_indices in entity_indices:
            idx = tok_indices[tok_indices >= 0]
            attn_scores = th.index_select(attn_scores_out,dim=0, index=idx)
            attn_weights = functional.softmax(attn_scores,dim=0)
            h_entity = th.matmul(attn_weights, blstm_out[idx])
            h_entity = h_entity.sum(axis=0)
            h_entities.append(h_entity)
        h_entities = th.cat(h_entities)
        return h_entities

    def _get_argsets_from_candidates(self, candidates):
        argsets = set()
        for argset_idx in candidates.get_combinations(r=2):
            key = tuple(sorted([self.idx_to_element[a[1].item()] for a in argset_idx]))
            if key in self.inverted_schema.keys():
                argset = tuple((c, candidates[c]) for c in argset_idx)
                argsets.add(argset)
        return argsets

    def _generate_to_predict(self, argsets):
        to_predict = []
        for argset in argsets:
            key = tuple(sorted([self.idx_to_element[arg[0][1].item()] for arg in argset]))
            if len(key) > 1:
                rels = self.inverted_schema[key]
                for rel in rels.keys():
                    prediction_candidate = PredictionCandidate(
                        rel,
                        self.element_embeddings(th.tensor(self.element_to_idx[rel])),
                        tuple([arg[0][1] for arg in argset]),
                        [arg[1] for arg in argset],
                    )
                    to_predict.append(prediction_candidate)
        return to_predict

    def forward(self, sentence, entity_names, entity_indices):

        embedded_sentence = self.word_embeddings(sentence)

        blstm_out, _ = self.blstm(
            embedded_sentence.view(embedded_sentence.shape[0], 1, -1)
        )

        h_entities = self.get_h_entities(entity_indices, blstm_out)

        # candidate_names = entity_names
        # candidate_hidden_vectors = h_entities

        candidates = ElementList()
        for name, hidden_vector in zip(entity_names, h_entities):
            el = Element(name, hidden_vector, None)
            candidates.add_element(el)

        new_candidates = True
        layer = 1
        predictions = []
        c = self.cell.init_cell_state()

        # while new_candidates and layer <= self.max_layer_height:
        #     # candidate_indices = th.arange(entity_names.shape[0])
        #     # tp_combinations = th.combinations(entity_indices,r=2)

        #     # for comb in tp_combinations:
        #     #     candidate_names.append(th.index_select(candidate_names,dim=0,index=comb))
        #     #     candidate_hidden_vectors.append(th.index_select(candidate_hidden_vectors,dim=0,index=comb))


        #     new_candidates = False
        #     argsets = self._get_argsets_from_candidates(candidates)
        #     to_predict = self._generate_to_predict(argsets)
        #     for tp in to_predict:
        #         e = tp.predicate_embedding
        #         hjs = tp.argument_hidden_vectors
        #         cjs = [c for _ in hjs]
        #         h, c = self.cell.forward(hjs, cjs, e)
        #         logits = self.output_linear(h)
        #         p = functional.softmax(logits, dim=0)
        #         if p[0] > 0.5:  # assumes 0 index is the positive class
        #             candidates.add_element(Element(th.tensor(th.tensor(self.element_to_idx[tp.predicate])), h, tp.argument_names))
        #             new_candidates = True
        #             predictions.append((p, th.tensor(self.element_to_idx[tp.predicate]), tp.argument_names))

        #     layer += 1
        return h_entities

def sent_to_idxs(sentence,vocab_dict):
    token_list = sentence.split()

    index_list = []
    for token in token_list:
        if token in vocab_dict:
            index_list.append(vocab_dict[token])
        else:
            index_list.append(vocab_dict["UNK"])
    return th.LongTensor(index_list)

def get_sentences(percent_test):
    with open("../data/text_sentences.txt") as f:
        sentences = f.read().splitlines()

    entities = json.load(open("../data/entity_labels.json", "r"))

    with open('../data/relation_labels.json','r') as f:
        relations = json.load(f)

    relations = [tuple((p, tuple(args)) for p, args in relation) for relation in relations]

    zipped_data = list(zip(sentences, entities, relations))

    # deletes sentences with no relations
    for i in range(len(zipped_data) - 1, -1, -1):
        if zipped_data[i][2] == []:
            del zipped_data[i]

    # print("new dataset length:", len(zipped_data))
    train_data, test_data = train_test_split(
        zipped_data, test_size=percent_test, random_state=0
    )

    return train_data, test_data


def main():
    train_data, test_data = get_sentences(0.2)

    vocab_dict = eval(open("../data/vocab_dict.txt", "r").read())
    config = BioInferTaskConfiguration().from_json("../data/configuration.json")
    element_to_idx = config.element_to_idx

    model = INNModel(
        vocab_dict,
        WORD_EMBEDDING_DIM,
        RELATION_EMBEDDING_DIM,
        HIDDEN_DIM,
        config.schema,
        config.inverted_schema,
        config.element_to_idx,
        5,
    )

    optimizer = th.optim.Adadelta(model.parameters(), lr=1.0)
    criterion = nn.NLLLoss()

    writer = SummaryWriter()

    sample = train_data[0]

    sent_idxs = sent_to_idxs(sample[0],vocab_dict)
    entity_names = th.tensor([th.tensor(element_to_idx[e[0]]) for e in sample[1]])
    entity_indices = [th.tensor(e[1]) for e in sample[1]]
    entity_indices = th.stack([functional.pad(e,pad=(0,5-len(e)),mode='constant',value=-1) for e in entity_indices])
    entity_names = entity_names.reshape(-1,1)

    writer.add_graph(model, input_to_model=(sent_idxs, entity_names, entity_indices), verbose=True)
    writer.flush()

    for epoch in range(EPOCHS):
        for step, x in enumerate(train_data):
            sent_idxs = sent_to_idxs(x[0],vocab_dict)
            entity_names = th.tensor([th.tensor(element_to_idx[e[0]]) for e in x[1]])
            entity_names = entity_names.reshape(-1,1)
            entity_indices = [th.tensor(e[1]) for e in x[1]]
            entity_indices = th.stack([functional.pad(e,pad=(0,5-len(e)),mode='constant',value=-1) for e in entity_indices])

            if all([len(e[1]) > 0 for e in x[1]]):
                optimizer.zero_grad()
                output = model(sent_idxs, entity_names, entity_indices)
                output_tuples = [(o[0], o[1].item(), tuple(i.item() for i in o[2])) for o in output]
                relations = x[2]
                relations = [(element_to_idx[r[0]], tuple(element_to_idx[i] for i in r[1])) for r in relations]

                loss = 0.0

                for true_rel in relations:
                    match = False
                    for prediction in output_tuples:
                        if true_rel == (prediction[1], prediction[2]):
                            loss += criterion(th.log(prediction[0].reshape(1,-1)), th.tensor([0]))
                            match = True
                    if not match:
                        loss += criterion(th.log(th.tensor([[0.001, 0.999]])), th.tensor([0]))

                for prediction in output_tuples:
                    if (prediction[1], prediction[2]) not in relations:
                        loss += criterion(th.log(prediction[0].reshape(1,-1)), th.tensor([1]))

                # y_vals = {}
                # keys = set(output).union(set(relations))
                # labels = []
                # predicted_probas = []
                # for key in keys:
                #     if key in relations:
                #         labels.append(0)
                #     else:
                #         labels.append(1)
                #     if key in output:
                #         predicted_probas.append(th.log(output[key].reshape(1, -1)))
                #     else:
                #         predicted_probas.append(th.tensor([0.0, 1.0]))


                # predicted_probas = th.log(th.cat(predicted_probas)).reshape(1,-1)



                # num actual relations - num predictions
                # num_rel_diff = len(relations) - len(output)

                # TODO: because these are dummy tensors, grad_fn doesn't exist; how to fix?
                # if there are fewer predictions than actual relations
                # give the model the maximum penalty * number of predictions it didn't make
                #if num_rel_diff > 0:
                #    dummy_label = 1  # TODO: Swapped; Check if this is correct
                #    dummy_pred = th.log(th.tensor([0.9999, 0.0001]))

                #    for _ in range(num_rel_diff):
                #        if label_list is None:
                #            label_list = [dummy_label]
                #            log_prob_list = dummy_pred
                #        else:
                #            label_list.append(dummy_label)
                #            log_prob_list = th.vstack((log_prob_list, dummy_pred))
                loss = Variable(loss, requires_grad=True)
                if loss != 0:
                    # loss.requires_grad = True  #TODO: the backpropagation wasn't working; setting it to True here
                    loss.backward()            #TODO: just makes an empty grad_fn() that doesn't update anything
                    optimizer.step()           #TODO: Note: backpropagation still doesn't work, prob with grad_fn
                print("Epoch {:05d} | Step {:05d} | Loss {:.4f} |".format(epoch, step, loss))

        val_acc = []
        for step, x in enumerate(test_data):
            with th.no_grad():
                sent_idxs = sent_to_idxs(x[0],vocab_dict)
                entities = [(th.tensor(element_to_idx[e[0]]), th.tensor(e[1])) for e in x[1]]
                output = model(sent_idxs, entities)
                relations = x[2]

                num_pred_correct = 0
                for prediction in output:
                    # changes the prediction's formatting so that it matches relations'
                    # formatted_prediction = [prediction[1], list(prediction[2])]
                    # formatted_prediction[0] = formatted_prediction[0].split('-')[1]

                    if prediction in relations:
                        num_pred_correct += 1

                val_acc.append(num_pred_correct / len(relations))

        print("Epoch {:05d} | Val Acc {:.4f} |".format(epoch, np.mean(val_acc)))



    # acc = float(th.sum(th.eq(batch.label, pred))) / len(batch.label)
    # print("Epoch {:05d} | Step {:05d} | Loss {:.4f} | Acc {:.4f} |".format(
    #     epoch, step, loss.item(), acc))

    writer.close()

    return 0


main()
