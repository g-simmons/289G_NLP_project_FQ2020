# this file is dedicated toward holding the actual model
import torch


def main():
    vocab_dict = eval(open('vocab_dict.txt', 'r').read())
    vocab_size = len(vocab_dict)

    dimension = 256  # from page 6 of the paper
    vocab_to_hidden_table = torch.nn.Embedding(vocab_size, dimension)

    # result = vocab_to_hidden_table(torch.LongTensor([0, 1])) <-- example of how to get index 0 and 1's embeddings

main()
