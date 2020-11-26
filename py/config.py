ENTITY_PREFIX = "e-"
PREDICATE_PREFIX = "p-"

EPOCHS = 3
BATCH_SIZE = 8
MAX_LAYERS = 2
MAX_ENTITY_TOKENS = 5

CELL_STATE_CLAMP_VAL = 1e4
HIDDEN_STATE_CLAMP_VAL = float('inf')

VECTOR_DIM = 256  # from page 6 of the paper
WORD_EMBEDDING_DIM = 256
HIDDEN_DIM = 256
RELATION_EMBEDDING_DIM = (
    256  # TODO: is the relation embedding dimension specified in the paper anywhere?
)
