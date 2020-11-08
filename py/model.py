# this file is dedicated toward holding the actual model
import torch
from torch import nn
from sklearn.model_selection import train_test_split

# NOTES FROM PAPER
# The final fully connected layer is 512 × 1024 × 2.
# We use Adadelta [31] as the optimizer with learning rate = 1.0.
# The batch size is 8.


class Model(nn.Module):
    def __init__(self, vocab_dict, embedding_dim, hidden_dim):
        super(Model, self).__init__()

        self.vocab_dict = vocab_dict        # vocabulary dictionary; converts tokens to indices
        self.vocab_size = len(vocab_dict)   # vocabulary size
        self.embedding_dim = embedding_dim  # size of embedding vector
        self.hidden_dim = hidden_dim        # size of lstm hidden vector; blstm's is 2x

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.blstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, num_layers=1)

    # accepts a sentence as input and returns a list of embedding vectors, one vector per token
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

    # used to feed input into the model and have the model produce output
    # TO BE EDITED WHEN MORE LAYERS ARE ADDED
    def forward(self, sentence):
        embedded_sentence = self.get_embeddings(sentence)
        # since bi-lstm's hidden vector is actually 2 concatenated vectors, it will be 2x as long (512)
        blstm_out, _ = self.blstm(embedded_sentence.view(embedded_sentence.shape[0], 1, -1))

        return blstm_out  # placeholder until we add more layers


def get_sentences(percent_test):
    with open('text_sentences.txt') as f:
        sentences = f.read().splitlines()

    training_sentences, test_sentences = train_test_split(sentences, test_size=percent_test, random_state=0)

    return training_sentences, test_sentences


def main():
    training_sentences, test_sentences = get_sentences(0.2)

    vocab_dict = eval(open('vocab_dict.txt', 'r').read())
    vector_dim = 256  # from page 6 of the paper

    the_model = Model(vocab_dict, vector_dim, vector_dim)
    print(the_model.forward(training_sentences[0]))  # example usage
    return 0


main()
