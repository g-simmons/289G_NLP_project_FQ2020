import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as functional

from config import *
from constants import *
from daglstmcell import DAGLSTMCell


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
        cell,
        cell_state_clamp_val,
        hidden_state_clamp_val,
    ):
        super().__init__()
        self.word_embedding_dim = word_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim

        self.word_embeddings = nn.Embedding(len(vocab_dict), self.word_embedding_dim)

        self.element_embeddings = nn.Embedding(
            len(element_to_idx.keys()), self.relation_embedding_dim
        )

        self.attn_scores = nn.Linear(
            in_features=self.word_embedding_dim * 2, out_features=1
        )

        self.blstm = nn.LSTM(
            input_size=self.word_embedding_dim,
            hidden_size=self.word_embedding_dim,
            bidirectional=True,
            num_layers=1,
        )

        self.cell = cell

        self.output_linear = nn.Sequential(
            nn.Linear(2 * self.word_embedding_dim, 4 * self.word_embedding_dim),
            nn.Linear(4 * self.word_embedding_dim, 2),
        )

    def get_h_entities(
        self, entity_indices, blstm_out, token_splits, H, curr_batch_size, is_entity
    ):
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
        attn_scores = torch.split(self.attn_scores(torch.cat(blstm_out)), token_splits)

        h_entities = []
        for sample in range(curr_batch_size):
            sample_entity_indices = [idx[idx >= 0] for idx in entity_indices[sample]]
            for idx in sample_entity_indices:
                sample_attn_scores = attn_scores[sample][idx.long()]
                sample_attn_weights = functional.softmax(sample_attn_scores, dim=0)
                blstm_vecs = blstm_out[sample][idx.long()]
                h_entity = torch.mul(sample_attn_weights, blstm_vecs).sum(axis=0)
                h_entities.append(h_entity)

        H_new[is_entity == 1] = torch.stack(h_entities)

        return H_new

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
        T,
        S,
        L,
        is_entity,
    ):
        wemb = self.word_embeddings(torch.cat(tokens))
        token_splits = [len(t) for t in tokens]
        embedded_tokens = torch.split(wemb, token_splits)
        curr_batch_size = len(entity_spans)
        blstm_out = [
            self.blstm(et.unsqueeze(0))[0].squeeze(0) for et in embedded_tokens
        ]

        # gets the hidden vector for each entity and stores them in H
        H = torch.randn(T.shape[0], self.word_embedding_dim * 2).to(self.device)
        # .detach()
        H = self.get_h_entities(
            entity_spans, blstm_out, token_splits, H, curr_batch_size, is_entity
        )
        # H.requires_grad_()

        C = torch.zeros(T.shape[0], self.word_embedding_dim * 2).to(self.device)
        # .detach().to(self.device)

        predictions = [
            PRED_TRUE if is_entity[i] == 1 else PRED_FALSE for i in range(0, T.shape[0])
        ]
        predictions = torch.stack(predictions).to(self.device)
        predictions.requires_grad_()

        for layer in torch.unique(L):
            if layer > 0:
                predictions = predictions.clone()
                parent_mask = self._get_parent_mask(L, layer, element_names, is_entity)
                # args_predicted = torch.all(torch.ge(predictions[S][:,:,1], 0.5),dim=1)
                # parent_mask = torch.logical_and(parent_mask,args_predicted)
                element_embeddings = self.element_embeddings(element_names[parent_mask])
                s = S[parent_mask, :]
                v = H[s]
                v = v.flatten(start_dim=1)
                cell_states = C[s]
                cell_states = cell_states.flatten(start_dim=1)
                h, c = self.cell.forward(v, cell_states, element_embeddings, s)
                H[parent_mask, :] = h
                C[parent_mask, :] = c
                logits = self.output_linear(H[parent_mask, :])
                sm_logits = functional.softmax(logits, dim=1)
                predictions[parent_mask, :] = sm_logits

        return predictions


class INNModelLightning(pl.LightningModule):
    def __init__(
        self,
        vocab_dict,
        element_to_idx,
        word_embedding_dim,
        cell_state_clamp_val,
        hidden_state_clamp_val,
    ):
        super().__init__()
        self.cell = DAGLSTMCell(
            blstm_out_dim=2 * word_embedding_dim,
            max_inputs=2,
            hidden_state_clamp_val=hidden_state_clamp_val,
            cell_state_clamp_val=cell_state_clamp_val,
        )
        self.inn = INNModel(
            vocab_dict=vocab_dict,
            element_to_idx=element_to_idx,
            word_embedding_dim=word_embedding_dim,
            relation_embedding_dim=2 * word_embedding_dim,
            cell=self.cell,
            cell_state_clamp_val=cell_state_clamp_val,
            hidden_state_clamp_val=hidden_state_clamp_val,
        )
        self.criterion = nn.NLLLoss()
        self.accuracy = pl.metrics.Accuracy()
        self.param_names = [p[0] for p in self.inn.named_parameters()]

    def forward(self, batch_sample):
        predictions = self.inn(*self.expand_batch(batch_sample))
        return predictions

    def training_step(self, batch_sample, batch_idx):
        self.logger.experiment.log(
            {"curr_batch_size": len(batch_sample["entity_spans"])}
        )
        opt = self.optimizers()
        raw_predictions = self.inn(*self.expand_batch(batch_sample))
        self.log('train_acc_step', self.accuracy(raw_predictions, batch_sample["labels"]))
        predictions = torch.log(raw_predictions)
        loss = self.criterion(predictions, batch_sample["labels"])
        predicted_pos = torch.sum(raw_predictions[:, 1] > 0.5)
        self.logger.experiment.log({"predicted_pos": predicted_pos})
        if predicted_pos > len(batch_sample["entity_spans"]):
            self.manual_backward(loss, opt)
            opt.step()
            self.logger.experiment.log({"loss": loss})

    def expand_batch(self, batch_sample):
        return (
            batch_sample["tokens"],
            batch_sample["entity_spans"],
            batch_sample["element_names"],
            batch_sample["T"],
            batch_sample["S"],
            batch_sample["L"],
            batch_sample["is_entity"],
        )

    def validation_step(self, batch_sample, batch_idx):
        raw_predictions = self.inn(*self.expand_batch(batch_sample))
        predictions = torch.log(raw_predictions).to(self.device)
        loss = self.criterion(predictions, batch_sample["labels"])
        # self.logger.experiment.log({"val_loss": loss})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=LEARNING_RATE)
        return optimizer
