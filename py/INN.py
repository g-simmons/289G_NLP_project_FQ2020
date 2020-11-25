import torch
from torch import nn
from torch.nn import functional as functional

from daglstmcell import DAGLSTMCell

from config import (
    ENTITY_PREFIX,
    PREDICATE_PREFIX,
    EPOCHS,
    WORD_EMBEDDING_DIM,
    VECTOR_DIM,
    HIDDEN_DIM,
    RELATION_EMBEDDING_DIM,
    BATCH_SIZE,
    MAX_LAYERS,
    MAX_ENTITY_TOKENS,
    CELL_STATE_CLAMP_VAL,
    HIDDEN_STATE_CLAMP_VAL
)


class INNModel(nn.Module):
    """INN model configuration.

    Parameters:
        vocab_dict (dict): The vocabulary for training, tokens to indices.
        word_embedding_dim (int): The size of the word embedding vectors.
        relation_embedding_dim (int): The size of the relation embedding vectors.
        hidden_dim (int): The size of LSTM hidden vector (effectively 1/2 of the desired BiLSTM output size).
        schema: The task schema
        element_to_idx (dict): dictionary mapping entity strings to unique integer values
    """

    def __init__(
        self,
        vocab_dict,
        element_to_idx,
        word_embedding_dim,
        relation_embedding_dim,
        hidden_dim,
        cell_state_clamp_val,
        hidden_state_clamp_val
    ):
        super().__init__()
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.relation_embedding_dim = relation_embedding_dim

        self.word_embeddings = nn.Embedding(len(vocab_dict), self.word_embedding_dim)

        self.element_embeddings = nn.Embedding(
            len(element_to_idx.keys()), self.relation_embedding_dim
        )

        self.attn_scores = nn.Linear(in_features=self.hidden_dim * 2, out_features=1)

        self.blstm = nn.LSTM(
            input_size=self.word_embedding_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True,
            num_layers=1,
        )

        self.cell = DAGLSTMCell(
            hidden_dim=2 * self.hidden_dim,
            relation_embedding_dim=self.relation_embedding_dim,
            max_inputs=2,
            hidden_state_clamp_val=hidden_state_clamp_val,
            cell_state_clamp_val=cell_state_clamp_val,
        )

        self.output_linear = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 4 * self.hidden_dim),
            nn.Linear(4 * self.hidden_dim, 2),
        )

    def get_h_entities(self, entity_indices, blstm_out, H):
        """Apply attention mechanism to entity representation.

        Args:
            entities (list of tuples::(str, (int,))): A list of pairs of an
                entity label and indices of words.
            blstm_out (torch.Tensor): The output hidden states of bi-LSTM.

        Returns:
            h_entities (torch.Tensor): The output hidden states of entity
                representation layer.

        """
        H_new = torch.clone(H)
        attn_scores_out = self.attn_scores(blstm_out)

        for i, tok_indices in enumerate(entity_indices):
            idx = tok_indices[tok_indices >= 0]

            if idx.nelement() == 0:
                # TODO: maybe set corresponding H_new entries to 0
                continue

            attn_scores = torch.index_select(attn_scores_out, dim=0, index=idx)
            attn_weights = functional.softmax(attn_scores, dim=0)

            # for each batch
            for batch_num in range(BATCH_SIZE):
                # gets the current batch's attention weights
                curr_batch_attn_weights = attn_weights[:, batch_num]

                # multiplies the current batch's attention weights and current batch's blstm_out
                # creates a T x D matrix where T is the number of tokens and D is blstm_out's dimension
                h_entity = torch.matmul(curr_batch_attn_weights, blstm_out[i, batch_num].unsqueeze(0))

                # creates a vector of length D (512) and stores it in H_new
                h_entity = h_entity.sum(axis=0)
                H_new[i, batch_num] = h_entity
        return H_new

    def _get_argsets_from_candidates(self, candidates):
        argsets = set()
        for argset_idx in candidates.get_combinations(r=2):
            key = tuple(sorted([self.idx_to_element[a[1].item()] for a in argset_idx]))
            if key in self.inverted_schema.keys():
                argset = tuple((c, candidates[c]) for c in argset_idx)
                argsets.add(argset)
        return argsets

    def _generate_to_predict(self, argsets):
        to_predict = []
        for argset in argsets:
            key = tuple(
                sorted([self.idx_to_element[arg[0][1].item()] for arg in argset])
            )
            if len(key) > 1:
                rels = self.inverted_schema[key]
                for rel in rels.keys():
                    prediction_candidate = PredictionCandidate(
                        rel,
                        self.element_embeddings(torch.tensor(self.element_to_idx[rel])),
                        tuple([arg[0][1] for arg in argset]),
                        [arg[1] for arg in argset],
                    )
                    to_predict.append(prediction_candidate)
        return to_predict

    def forward(self, tokens, entity_spans, element_names, H, A, T, S):

        embedded_sentence = self.word_embeddings(tokens)

        print("shape 1", embedded_sentence.shape)
        blstm_out, _ = self.blstm(embedded_sentence)

        print("shape 2", blstm_out.shape)

        H = self.get_h_entities(entity_spans, blstm_out, H)

        # predictions = torch.empty((H.shape[0], 2),requires_grad=True)

        # predictions[0 : len(entity_spans)] = torch.tensor(
        #     [0.001, 0.999]
        # )  # predict positive for the entities

        predictions = []

        for i in range(0,len(entity_spans)):
            predictions.append(torch.tensor([0.001, 0.999]))

        c = self.cell.init_cell_state()

        for argset, target_idx, element_name in zip(S, T, element_names):
            if target_idx >= len(entity_spans) and element_name > -1:
                args_idx = argset[argset > -1]
                if torch.all(torch.stack(predictions)[args_idx, 1] > 0.5):
                    hidden_vectors = H[args_idx]
                    cell_states = [c for _ in hidden_vectors]
                    e = self.element_embeddings(element_name)
                    h_out, c = self.cell.forward(hidden_vectors, cell_states, e)
                    H[target_idx] = h_out
                    logits = self.output_linear(h_out)
                    # predictions[target_idx] = functional.softmax(logits, dim=0)
                    sm_logits = functional.softmax(logits, dim=0)
                    predictions.append(sm_logits)
                else:
                    predictions.append(torch.tensor([0.999, 0.001]))
                    predictions[target_idx] = torch.tensor(
                        [0.999, 0.001]
                    )  # predict negative if all arguments have not been predicted positive
        predictions = torch.stack(predictions)

        return predictions
