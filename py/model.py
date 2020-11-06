# this file is detected toward holding the actual model
import torch.nn as nn

# may not warrant a function
def create_vocab_to_hidden_table(dimension):
    vocab_dict = eval(open('vocab_dict.txt', 'r').read())

    vocab_size = len(vocab_dict)
    return nn.Embedding(vocab_size, dimension)

create_vocab_to_hidden_table(10)
