import sys
sys.path.append('../lib/BioInfer_software_1.0.1_Python3/')
from BIParser import BIParser

# reads in the BioInfer corpus, creates its vocabulary, and converts the vocabulary into a dictionary
# dictionary is stored in a file called vocab_dict.txt
# dictionary includes "UNK", which represents unknown tokens
def create_vocab_dictionary():
    parser = BIParser()
    with open('../data/BioInfer_corpus_1.1.1.xml', 'r') as f:
        parser.parse(f)

    vocab = {u'UNK'}

    for s in parser.bioinfer.sentences.sentences:
        for token in s.tokens:
            vocab.add(token.getText())

    vocab_size = len(vocab)
    vocab_index_list = [index for index in range(vocab_size)]

    vocab_dict = dict(zip(vocab, vocab_index_list))
    with open('vocab_dict.txt', 'w') as file:
        file.write(str(vocab_dict))


# reads in the BioInfer corpus and stores its sentences in a text file called text_sentences.txt
# each sentence is on its own line; file is used as input for the model
def create_text_sentences():
    parser = BIParser()
    with open('../data/BioInfer_corpus_1.1.1.xml', 'r') as f:
        parser.parse(f)

    with open('text_sentences.txt', 'w') as file:
        for s in parser.bioinfer.sentences.sentences:
            file.write(str(s.getText()) + '\n')


create_vocab_dictionary()
create_text_sentences()
