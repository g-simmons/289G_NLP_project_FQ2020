ENTITY_PREFIX = "e-"
PREDICATE_PREFIX = "p-"

EPOCHS = 3
BATCH_SIZE = 4
MAX_LAYERS = 2
MAX_ENTITY_TOKENS = 5
LEARNING_RATE = 0.1

CELL_STATE_CLAMP_VAL = 1e4
HIDDEN_STATE_CLAMP_VAL = float("inf")

PREPPED_DATA_PATH = "../data/prepped_dataset.pickle"
XML_PATH = "../data/BioInfer_corpus_1.1.1.xml"

VECTOR_DIM = 256  # from page 6 of the paper
WORD_EMBEDDING_DIM = 256
HIDDEN_DIM = 256
RELATION_EMBEDDING_DIM = (
    256  # TODO: is the relation embedding dimension specified in the paper anywhere?
)

EXCLUDE_SAMPLES = [
    681,  # has no entities
]
