import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.tensorboard import SummaryWriter

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
    HIDDEN_STATE_CLAMP_VAL,
    LEARNING_RATE,
)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pytorch_lightning as pl
class INNModel(pl.LightningModule):
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
        cell,
        cell_state_clamp_val,
        hidden_state_clamp_val,
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

        self.cell = cell

        self.output_linear = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 4 * self.hidden_dim),
            nn.Linear(4 * self.hidden_dim, 2),
        )

    def get_h_entities(self, entity_indices, blstm_out, H, curr_batch_size):
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

            # for each batch entry
            for batch_entry_num in range(curr_batch_size):
                # gets the current batch entry's attention weights
                curr_batch_attn_weights = attn_weights[:, batch_entry_num]

                # multiplies the current batch entry's attention weights and current batch's blstm_out
                # creates a T x D matrix where T is the number of tokens and D is blstm_out's dimension
                h_entity = torch.matmul(
                    curr_batch_attn_weights, blstm_out[i, batch_entry_num].unsqueeze(0)
                )

                # creates a vector of length D (512) and stores it in H_new
                h_entity = h_entity.sum(axis=0)
                H_new[i, batch_entry_num] = h_entity

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

    def forward(
        self, tokens, entity_spans, element_names, T, S, entity_spans_size, tokens_size
    ):
        curr_batch_size = entity_spans.shape[1]

        # gets the embedding for each token
        embedded_sentence = self.word_embeddings(tokens)
        # to make computation faster, gets rid of padding by packing the batch tensor
        # only RNN can use packed tensors
        embedded_sentence = pack_padded_sequence(embedded_sentence, tokens_size.cpu())
        blstm_out, _ = self.blstm(embedded_sentence)
        # unpacks the output tensor (re-adds the padding) so that other functions can use it
        blstm_out, _ = pad_packed_sequence(blstm_out)
        # gets the hidden vector for each entity and stores them in H
        H = (
            torch.randn(T.shape[0], curr_batch_size, 2 * HIDDEN_DIM)
            .detach()
            .to(self.device)
        )
        H = self.get_h_entities(entity_spans, blstm_out, H, curr_batch_size)

        predictions = []

        # for each batch entry
        for batch_entry_num in range(curr_batch_size):
            predictions_row = []
            # for each entity span for the current batch entry
            for _ in range(entity_spans_size[batch_entry_num]):
                # add a "prediction" that's basically certain it's right
                predictions_row.append(torch.tensor([0.001, 0.999]).to(self.device))
            predictions.append(predictions_row)
        c = self.cell.init_cell_state()

        # for each batch entry
        for batch_entry_num in range(curr_batch_size):
            # iterates over the current batch entry's S, T, and element_names
            # and generates the current batch entry's predictions
            for argset, target_idx, element_name in zip(
                S[:, batch_entry_num],
                T[:, batch_entry_num],
                element_names[:, batch_entry_num],
            ):

                if (
                    target_idx >= entity_spans_size[batch_entry_num]
                    and element_name > -1
                ):
                    args_idx = argset[argset > -1]
                    stacked = torch.stack(predictions[batch_entry_num])
                    if torch.all(stacked[args_idx, 1] > 0.5):
                        hidden_vectors = H[args_idx, batch_entry_num]
                        cell_states = [c for _ in hidden_vectors]
                        e = self.element_embeddings(element_name)
                        cell_states = (
                            torch.cat(cell_states).unsqueeze(1).to(self.device)
                        )
                        h_out, c = self.cell.forward(hidden_vectors, cell_states, e)
                        h_out = h_out.to(self.device)
                        c = c.to(self.device)
                        H[target_idx, batch_entry_num] = h_out

                        logits = self.output_linear(h_out)
                        sm_logits = functional.softmax(logits, dim=0)

                        predictions[batch_entry_num].append(sm_logits)

                    else:
                        predictions[batch_entry_num].append(
                            torch.tensor([0.999, 0.001]).to(self.device)
                        )
                        predictions[batch_entry_num][target_idx] = torch.tensor(
                            [0.999, 0.001]
                        ).to(
                            self.device
                        )  # predict negative if all arguments have not been predicted positive
            # concatenates the batch entry's predictions along the 0 dimension
            predictions[batch_entry_num] = torch.stack(
                predictions[batch_entry_num], dim=0
            )

        # concatenates all predictions along the 0 dimension; basically a list of predictions
        # expected to have shape N x 2, where N is the number of predictions
        predictions = torch.cat(predictions, dim=0)
        print(predictions)
        predictions.clamp_(min=1e-3)

        return predictions


class INNModelLightning(pl.LightningModule):
    def __init__(
        self,
        vocab_dict,
        element_to_idx,
        word_embedding_dim,
        relation_embedding_dim,
        hidden_dim,
        cell_state_clamp_val,
        hidden_state_clamp_val,
    ):
        super().__init__()
        self.cell = DAGLSTMCell(
            hidden_dim=2 * hidden_dim,
            relation_embedding_dim=relation_embedding_dim,
            max_inputs=2,
            hidden_state_clamp_val=hidden_state_clamp_val,
            cell_state_clamp_val=cell_state_clamp_val,
        )
        self.inn = INNModel(
            vocab_dict=vocab_dict,
            element_to_idx=element_to_idx,
            word_embedding_dim=word_embedding_dim,
            relation_embedding_dim=relation_embedding_dim,
            hidden_dim=hidden_dim,
            cell=self.cell,
            cell_state_clamp_val=cell_state_clamp_val,
            hidden_state_clamp_val=hidden_state_clamp_val,
        )
        self.criterion = nn.NLLLoss()
        self.param_names = [p[0] for p in self.inn.named_parameters()]

    def forward(self, batch_sample):
        predictions = self.inn(
            batch_sample["tokens"],
            batch_sample["entity_spans"],
            batch_sample["element_names"],
            batch_sample["H"],
            batch_sample["T"],
            batch_sample["S"],
            batch_sample["entity_spans_pre-padded_size"],
            batch_sample["tokens_pre-padded_size"],
        )
        return predictions

    def training_step(self, batch_sample, batch_idx):
        self.logger.experiment.log(
            {"curr_batch_size": batch_sample["entity_spans"].shape[1]}
        )
        opt = self.optimizers()
        raw_predictions = self.inn(
            batch_sample["tokens"],
            batch_sample["entity_spans"],
            batch_sample["element_names"],
            batch_sample["T"],
            batch_sample["S"],
            batch_sample["entity_spans_pre-padded_size"],
            batch_sample["tokens_pre-padded_size"],
        )
        predictions = torch.log(raw_predictions)
        loss = self.criterion(predictions, batch_sample["labels"])
        predicted_pos = torch.sum(raw_predictions[:, 1] > 0.5)
        self.logger.experiment.log({"predicted_pos": predicted_pos})
        if len(predictions) > len(batch_sample["entity_spans"]):
            self.manual_backward(loss, opt)
            opt.step()
            self.logger.experiment.log({"loss": loss})

    def validation_step(self, batch_sample, batch_idx):
        raw_predictions = self.inn(
            batch_sample["tokens"],
            batch_sample["entity_spans"],
            batch_sample["element_names"],
            batch_sample["T"],
            batch_sample["S"],
            batch_sample["entity_spans_pre-padded_size"],
            batch_sample["tokens_pre-padded_size"],
        )
        predictions = torch.log(raw_predictions).to(self.device)
        loss = self.criterion(predictions, batch_sample["labels"])
        self.logger.experiment.log({"val_loss": loss})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=LEARNING_RATE)
        return optimizer
