ENTITY_PREFIX = "e-"
PREDICATE_PREFIX = "p-"

EPOCHS = 5                      #changed for BERT
BATCH_SIZE = 16
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
VAL_SHUFFLE = False
MAX_LAYERS = 2
MAX_ENTITY_TOKENS = 5
LEARNING_RATE = 0.01                #changed for BERT

# use one of theses file names for training from scratch use any of these
#prepped_dataset_bioBert.pickle
#prepped_dataset_sciBert.pickle
#prepped_dataset_Bert.pickle
PREPPED_DATA_PATH = "../data/prepped_dataset_bioBert.pickle"
BioBERT_DATA_PATH = "../data/biobert_v1.1_pubmed/vocab.txt"
XML_PATH = "../data/BioInfer_corpus_1.1.1.xml"

WORD_EMBEDDING_DIM = 768                        #changed for BERT
BLSTM_OUT_DIM = 2 * WORD_EMBEDDING_DIM
HIDDEN_DIM_BERT = 768
# MASK_PREVIOUS_ARGS = False

EXCLUDE_SAMPLES = [
    681,  # has no entities
]

# options are bert, from-scratch, sci-bert, bio-bert
FREEZE_BERT_EPOCH = 2
ENCODING_METHOD = "sci-bert"                #changed for BERT

VAL_DIMS = False
