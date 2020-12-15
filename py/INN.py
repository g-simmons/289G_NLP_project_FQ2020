import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as functional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import *

from config import *
from constants import *
from daglstmcell import DAGLSTMCell


class FromScratchEncoder(pl.LightningModule):
    def __init__(self, vocab_dict, word_embedding_dim):
        super().__init__()
        self.word_embedding_dim = word_embedding_dim
        self.vocab_dict = vocab_dict
        self.word_embeddings = nn.Embedding(
            len(self.vocab_dict), self.word_embedding_dim
        )
        self.blstm = nn.LSTM(
            input_size=self.word_embedding_dim,
            hidden_size=self.word_embedding_dim,
            bidirectional=True,
            num_layers=1,
        )

    def forward(self, tokens):
        wemb = self.word_embeddings(torch.cat(tokens))
        token_splits = [len(t) for t in tokens]
        embedded_tokens = torch.split(wemb, token_splits)
        blstm_out = [
            self.blstm(et.unsqueeze(0))[0].squeeze(0) for et in embedded_tokens
        ]
        return blstm_out, token_splits


class BERTEncoder(pl.LightningModule):
    def __init__(self, output_bert_hidden_states):
        super().__init__()
        self.bert_config = AutoConfig.from_pretrained(
            "allenai/scibert_scivocab_uncased"
        )
        self.output_bert_hidden_states = output_bert_hidden_states
        if self.output_bert_hidden_states:
            self.bert_config.output_hidden_states = output_bert_hidden_states
        self.bert = AutoModel.from_pretrained(
            "allenai/scibert_scivocab_uncased", config=self.bert_config
        )

    def forward(self, bert_tokens, mask):
        tokens = bert_tokens
        tokens = tokens.squeeze(0)
        mask = mask.squeeze(0)

        token_splits = [
            len(t) for t in tokens
        ]  # should be tokens or bert_tokens? check len matches later on

        bert_outs = []
        for toks in tokens:
            bert_out = self.bert(tokens, attention_mask=mask)[0]
            bert_out = bert_out.permute(1, 0, 2)
            bert_outs.append(bert_out)

        return bert_outs, token_splits


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
        encoding_method,
        output_bert_hidden_states,
        word_embedding_dim,
        relation_embedding_dim,
        hidden_dim_bert,
        cell,
        cell_state_clamp_val,
        hidden_state_clamp_val,
    ):
        super().__init__()
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim_bert = hidden_dim_bert
        self.relation_embedding_dim = relation_embedding_dim
        self.encoding_method = encoding_method
        self.output_bert_hidden_states = output_bert_hidden_states

        if self.encoding_method == "bert":
            self.encoder = BERTEncoder(self.output_bert_hidden_states)
            linear_in_dim = self.hidden_dim_bert
        else:
            self.encoder = FromScratchEncoder(vocab_dict, word_embedding_dim)
            linear_in_dim = 2 * self.word_embedding_dim

        self.element_embeddings = nn.Embedding(
            len(element_to_idx.keys()), self.relation_embedding_dim
        )

        self.attn_scores = nn.Linear(in_features=linear_in_dim, out_features=1)

        self.cell = cell

        self.output_linear = nn.Sequential(
            nn.Linear(linear_in_dim, 4 * self.word_embedding_dim),
            nn.Linear(4 * self.word_embedding_dim, 2),
        )

    def get_h_entities(
        self, entity_indices, encoding_out, token_splits, H, curr_batch_size, is_entity
    ):
        """Apply attention mechanism to entity representation.
        Args:
            entities (list of tuples::(str, (int,))): A list of pairs of an
                entity label and indices of words.
            encoding_out (torch.Tensor): The output hidden states of encoding module.
        Returns:
            h_entities (torch.Tensor): The output hidden states of entity
                representation layer.
        """
        H_new = torch.clone(H)
        attn_scores = torch.split(
            self.attn_scores(torch.cat(encoding_out)), token_splits
        )

        h_entities = []
        for sample in range(curr_batch_size):
            sample_entity_indices = [idx[idx >= 0] for idx in entity_indices[sample]]
            for idx in sample_entity_indices:
                sample_attn_scores = attn_scores[sample][idx.long()]
                sample_attn_weights = functional.softmax(sample_attn_scores, dim=0)
                encoding_vecs = encoding_out[sample][idx.long()]
                h_entity = torch.mul(sample_attn_weights, encoding_vecs).sum(axis=0)
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
        bert_tokens,
        mask,
        entity_spans,
        element_names,
        T,
        S,
        L,
        is_entity,
    ):
        encoding_out = None
        if self.encoding_method == "bert":
            encoding_out, token_splits = self.encoder(bert_tokens, mask)
        elif self.encoding_method == "from-scratch":
            encoding_out, token_splits = self.encoder(tokens)
        if not encoding_out:
            raise ValueError("encoding did not occur check encoding_method")

        curr_batch_size = len(entity_spans)

        # gets the hidden vector for each entity and stores them in H
        H = (
            torch.randn(T.shape[0], self.word_embedding_dim * 2)
            .detach()
            .to(self.device)
        )
        H = self.get_h_entities(
            entity_spans, encoding_out, token_splits, H, curr_batch_size, is_entity
        )

        C = (
            torch.zeros(T.shape[0], self.word_embedding_dim * 2)
            .detach()
            .to(self.device)
        )

        predictions = [
            PRED_TRUE if is_entity[i] == 1 else PRED_FALSE for i in range(0, T.shape[0])
        ]
        predictions = torch.stack(predictions).to(self.device)
        predictions.requires_grad_()

        for layer in torch.unique(L):
            if layer > 0:
                predictions = predictions.clone()
                parent_mask = self._get_parent_mask(L, layer, element_names, is_entity)
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
        encoding_method,
        output_bert_hidden_states,
        word_embedding_dim,
        hidden_dim_bert,
        cell_state_clamp_val,
        hidden_state_clamp_val,
    ):
        super().__init__()
        self.encoding_method = encoding_method
        if self.encoding_method == "bert":
            encoding_dim = hidden_dim_bert
        else:
            encoding_dim = 2 * word_embedding_dim

        self.cell = DAGLSTMCell(
            encoding_dim=encoding_dim,
            max_inputs=2,
            hidden_state_clamp_val=hidden_state_clamp_val,
            cell_state_clamp_val=cell_state_clamp_val,
        )

        self.inn = INNModel(
            vocab_dict=vocab_dict,
            element_to_idx=element_to_idx,
            encoding_method=encoding_method,
            output_bert_hidden_states=output_bert_hidden_states,
            hidden_dim_bert=hidden_dim_bert,
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
        self.log(
            "train_acc_step", self.accuracy(raw_predictions, batch_sample["labels"])
        )
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
            batch_sample["from_scratch_tokens"],
            batch_sample["bert_tokens"],
            batch_sample["mask"],
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
