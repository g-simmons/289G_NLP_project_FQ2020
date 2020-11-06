# this file is dedicated toward holding the actual model
import torch.nn as nn


def main():
    vocab_dict = eval(open('vocab_dict.txt', 'r').read())
    vocab_size = len(vocab_dict)

    dimension = 256  # from page 6 of the paper
    vocab_to_hidden_table = nn.Embedding(vocab_size, dimension)

    # vocab_to_hidden_table([0, 1])  <-- this is an example of how to get index 0 and 1's embeddings

main()
