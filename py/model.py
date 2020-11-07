# this file is dedicated toward holding the actual model
import torch

vocab_dict = eval(open('vocab_dict.txt', 'r').read())
vocab_size = len(vocab_dict)

hidden_vec_dim = 256  # from page 6 of the paper
vocab_to_hidden_table = torch.nn.Embedding(vocab_size, hidden_vec_dim)


# accepts a list of tokens as input and returns a list of embedding vectors, one vector per token
# i-th embedding vector corresponds to the i-th token
def get_embeddings(token_list):
    index_list = []
    for token in token_list:
        # if the token exists in the vocabulary
        if token in vocab_dict:
            index_list.append(vocab_dict[token])

        # if the token does not exist in the vocabulary
        else:
            index_list.append(vocab_dict[u'UNK'])

    return vocab_to_hidden_table(torch.LongTensor(index_list))


def main():
    print(get_embeddings(['the', 'a', 'b']))  # example usage of function

    return 0


main()
