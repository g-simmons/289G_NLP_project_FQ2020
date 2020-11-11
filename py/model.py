# -*- coding: utf-8 -*-
"""The model configuration of Iterative Neural Network (INN).

This module implements the INN model proposed by Cao (2019), Nested Relation
Extraction with Iterative Neural Network.

Example:

        $ python model.py

Authors:
    Sammy Jia - https://github.com/sajia28
    Fangzhou Li - https://github.com/fangzhouli

TODO:
    *

"""

# this file is dedicated toward holding the actual model
from os import path
import pickle
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

# NOTES FROM PAPER
# The final fully connected layer is 512 × 1024 × 2.
# We use Adadelta [31] as the optimizer with learning rate = 1.0.
# The batch size is 8.


class INNModel(nn.Module):
    """INN model configuration.

    Parameters:
        vocab_dict (dict): The vocabulary for training, tokens to indices.
        embedding_dim (int): The size of the embedding vector.
        hidden_dim (int): The size of LSTM hidden vector, if bi-LSTM,
            multilplied by 2

    """

    def __init__(self, vocab_dict, embedding_dim, hidden_dim):
        super().__init__()
        self.vocab_dict = vocab_dict
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # self.entity_labels = entity_labels

        self.embedding_layer = nn.Embedding(
            len(vocab_dict),
            self.embedding_dim)
        # For each hidden
        self.attn_scores = nn.Linear(
            in_features=self.hidden_dim * 2,
            out_features=1)
        self.blstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            num_layers=1)

    # Accepts a sentence as input and returns a list of embedding vectors, one
    #   vector per token.
    # i-th embedding vector corresponds to the i-th token
    def get_embeddings(self, sentence):
        token_list = sentence.split()

        index_list = []
        for token in token_list:
            # if the token exists in the vocabulary
            if token in self.vocab_dict:
                index_list.append(self.vocab_dict[token])

            # if the token does not exist in the vocabulary
            else:
                index_list.append(self.vocab_dict[u'UNK'])

        return self.embedding_layer(torch.LongTensor(index_list))

    def get_h_entities(self, entities, blstm_out):
        """Apply attention mechanism to entity representation.

        Args:
            entities (list of tuples::(str, (int,))): A list of pairs of an
                entity label and indices of words.
            blstm_out (torch.Tensor): The output hidden states of bi-LSTM.

        Returns:
            h_entities (torch.Tensor): The output hidden states of entity
                representation layer.

        """
        h_entities = []
        attn_scores_out = self.attn_scores(blstm_out)
        for entity in entities:
            h_entity = 0
            attn_weights = F.softmax(attn_scores_out[list(entity[1])])
            for i in range(len(attn_weights)):
                h_entity += attn_weights[i] * blstm_out[list(entity[1])][i]
            h_entities.append(h_entity)
        h_entities = torch.cat(h_entities)
        return h_entities

    # used to feed input into the model and have the model produce output
    # TO BE EDITED WHEN MORE LAYERS ARE ADDED
    # TODO
    #   * Dataset instead a single sentence.
    #   * Does input entities enough?
    def forward(self, x):
        sentence, entities = x
        embedded_sentence = self.get_embeddings(sentence)
        # since bi-lstm's hidden vector is actually 2 concatenated vectors,
        #   it will be 2x as long (512)
        blstm_out, _ = self.blstm(embedded_sentence.view(
            embedded_sentence.shape[0], 1, -1))
        h_entities = self.get_h_entities(entities, blstm_out)
        return h_entities  # placeholder until we add more layers


def get_sentences(percent_test):
    with open('text_sentences.txt') as f:
        sentences = f.read().splitlines()
    entities = pickle.load(
        open(path.abspath(path.dirname(__file__)) + "/entities.pickle", 'rb'))

    train_data, test_data = train_test_split(
        list(zip(sentences, entities)), test_size=percent_test, random_state=0)

    return train_data, test_data


def main():
    train_data, test_data = get_sentences(0.2)

    vocab_dict = eval(open('vocab_dict.txt', 'r').read())
    vector_dim = 256  # from page 6 of the paper

    model = INNModel(vocab_dict, vector_dim, vector_dim)

    # example usage
    print(train_data[0][0])
    output = model.forward(train_data[0])
    # print(len(train_data[0]))
    # print(output.size())
    return 0


main()
