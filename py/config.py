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

CELL_STATE_CLAMP_VAL = 1e4
HIDDEN_STATE_CLAMP_VAL = 1e6

PREPPED_DATA_PATH = "../data/prepped_dataset.pickle"
XML_PATH = "../data/BioInfer_corpus_1.1.1.xml"

WORD_EMBEDDING_DIM = 256                        #changed for BERT
BLSTM_OUT_DIM = 2 * WORD_EMBEDDING_DIM
HIDDEN_DIM_BERT = 768
# MASK_PREVIOUS_ARGS = False

EXCLUDE_SAMPLES = [
    681,  # has no entities
]

# options are bert, from-scratch
FREEZE_BERT_EPOCH = 2
ENCODING_METHOD = "from-scratch"                #changed for BERT

VAL_DIMS = False
