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

    def get_h_entities(self, entity_indices, blstm_out, token_splits, H, curr_batch_size, is_entity):
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
        attn_scores = torch.split(self.attn_scores(torch.cat(blstm_out)),token_splits)

        h_entities = []
        for sample in range(curr_batch_size):
            sample_entity_indices = [idx[idx >= 0] for idx in entity_indices[sample]]
            for idx in sample_entity_indices:
                sample_attn_scores = attn_scores[sample][idx]
                sample_attn_weights = functional.softmax(sample_attn_scores, dim=0)
                blstm_vecs = blstm_out[sample][idx]
                h_entity = torch.mul(sample_attn_weights,blstm_vecs).sum(axis=0)
                h_entities.append(h_entity)

        H_new[is_entity == 1] = torch.stack(h_entities)

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

    def _get_parent_mask(self, L, layer, element_names, is_entity):
        mask = L == layer
        mask = torch.logical_and(mask, element_names > -1)
        mask = torch.logical_and(mask, torch.logical_not(is_entity))
        return mask

    def forward(
        self,
        tokens,
        entity_spans,
        element_names,
        A,
        T,
        S,
        L,
        is_entity,
    ):
        wemb = self.word_embeddings(torch.cat(tokens))
        token_splits = [len(t) for t in tokens]
        embedded_tokens = torch.split(wemb,token_splits)
        curr_batch_size = len(entity_spans)
        blstm_out = [self.blstm(et.unsqueeze(0))[0].squeeze(0) for et in embedded_tokens]

        # gets the hidden vector for each entity and stores them in H
        _, candidate_splits = torch.unique_consecutive(A,return_counts=True)
        H = torch.randn(T.shape[0], 2 * HIDDEN_DIM).detach().to(self.device)
        H = self.get_h_entities(entity_spans, blstm_out, token_splits, H, curr_batch_size, is_entity)

        C = (
            torch.zeros(T.shape[0], 2 * HIDDEN_DIM)
            .clone()
            .detach()
            .requires_grad_(True)
        )
        predictions = torch.stack([
            torch.tensor([0.999, 0.001]) for _ in range(T.shape[0])
        ])  # default false

        # predict true for entities
        predictions[is_entity == 1] = torch.tensor([0.001, 0.999])

        for layer in torch.unique(L):
            if layer > 0:
                parent_mask = self._get_parent_mask(L, layer, element_names, is_entity)
                element_embeddings = self.element_embeddings(element_names[parent_mask])
                H, C = self.cell.forward(
                    H, C, element_embeddings, S[parent_mask,:]
                )
                logits = self.output_linear(h_out)
                sm_logits = functional.softmax(logits, dim=0)
                predictions[mask] = sm_logits

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
            batch_sample["A"],
            batch_sample["T"],
            batch_sample["S"],
            batch_sample["L"],
            batch_sample["is_entity"],
        )
        return predictions

    def training_step(self, batch_sample, batch_idx):
        self.logger.experiment.log(
            {"curr_batch_size": len(batch_sample["entity_spans"])}
        )
        opt = self.optimizers()
        raw_predictions = self.inn(
            batch_sample["tokens"],
            batch_sample["entity_spans"],
            batch_sample["element_names"],
            batch_sample["A"],
            batch_sample["T"],
            batch_sample["S"],
            batch_sample["L"],
            batch_sample["is_entity"],
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
            batch_sample["A"],
            batch_sample["T"],
            batch_sample["S"],
            batch_sample["L"],
            batch_sample["is_entity"],
        )
        predictions = torch.log(raw_predictions).to(self.device)
        loss = self.criterion(predictions, batch_sample["labels"])
        self.logger.experiment.log({"val_loss": loss})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=LEARNING_RATE)
        return optimizer
