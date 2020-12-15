# 
"""
"""

from os import path
import pandas as pd
from transformers import AutoTokenizer

def load_corpus():
    """
    """
    path_data = path.abspath(path.dirname(__file__)) + \
        "/../data/text_sentences.txt"
    data = pd.read_csv(path_data, delimiter='\n', header=None)
    return data

def parse_bert(seq_original, seq_bert):
    """
    """

    def remove_leading_pounds(token):
        """
        """
        token_new = ''
        for c in token:
            if c != '#':
                token_new += c
        return token_new

    # Remove header and footer tags.
    seq_bert = seq_bert[1:-1]

    # Iterate over the original sequence and detect splitted tokens.
    mapped_indices_list = []
    j = 0
    for i in range(len(seq_original)):
        if seq_original[i] == seq_bert[j]:  # Not splitted.
            j += 1
            continue
        else:  # Detect splitted tokens.
            start = 0
            token_splitted = seq_original[i]
            token_mapping = remove_leading_pounds(seq_bert[j])
            mapped_indices = []
            while token_mapping == \
                    token_splitted[start : start + len(token_mapping)]:
                mapped_indices.append(j)
                start += len(token_mapping)
                j += 1
                token_mapping = remove_leading_pounds(seq_bert[j])
            mapped_indices_list.append((i, mapped_indices))
    return mapped_indices_list

if __name__ == '__main__':
    data = load_corpus()
    tokenizer = AutoTokenizer.from_pretrained(
        'allenai/scibert_scivocab_uncased')

    text_test = data[0][4]
    seq_original = [w.lower() for w in data[0][4].split(' ')]
    seq_bert = tokenizer.tokenize('[CLS] ' + text_test + ' [SEP]')
    print("Original word sequence:", seq_original)
    print("BERT word sequence:", seq_bert)

    splits = parse_bert(seq_original, seq_bert)
    for split in splits:
        print("Splitted token:", seq_original[split[0]])
        print("Splitted to:",
            ', '.join([seq_bert[1 : -1][i] for i in split[1]]))
        print()
