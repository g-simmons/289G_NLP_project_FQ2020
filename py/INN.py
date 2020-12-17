import pytorch_lightning as pl
from pytorch_lightning import metrics
import torch
from torch import nn
from torch.nn import functional as functional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import *

from config import *
from constants import *
from daglstmcell import DAGLSTMCell

import re

def update_batch_S(new_batch, batch):
    S = batch[0]["S"].clone()
    for i in range(1, len(batch)):
        s_new = batch[i]["S"].clone()
        s_new[s_new > -1] += batch[i - 1]["T"].shape[0]
        S = torch.cat([S, s_new])
    new_batch["S"] = S
    return new_batch


def collate_list_keys(new_batch, batch, list_keys):
    for key in list_keys:
        new_batch[key] = [sample[key] for sample in batch]
    return new_batch


def collate_cat_keys(new_batch, batch, cat_keys):
    for key in cat_keys:
        new_batch[key] = torch.cat([sample[key] for sample in batch])
    return new_batch


def collate_func(batch):
    cat_keys = ["element_names", "L", "labels", "is_entity"]
    list_keys = ["from_scratch_tokens", "bert_tokens", "entity_spans", "mask", "text"]

    if type(batch) == dict:
        batch = [batch]

    new_batch = {}
    new_batch = collate_list_keys(new_batch, batch, list_keys)
    new_batch = collate_cat_keys(new_batch, batch, cat_keys)
    new_batch = update_batch_S(new_batch, batch)

    T = torch.arange(len(new_batch["element_names"]))
    new_batch["T"] = T

    return new_batch


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
    def __init__(self, output_bert_hidden_states, freeze_bert_epoch):
        super().__init__()
        self.freeze_bert_epoch = freeze_bert_epoch
        print('loading pretrained BERT...')
        self.bert_config = AutoConfig.from_pretrained(
            "allenai/scibert_scivocab_uncased"
        )
        self.output_bert_hidden_states = output_bert_hidden_states
        if self.output_bert_hidden_states:
            self.bert_config.output_hidden_states = output_bert_hidden_states
        self.bert = AutoModel.from_pretrained(
            "allenai/scibert_scivocab_uncased", config=self.bert_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

    @staticmethod
    def parse_bert(seq_original, offset_mapping):
        word_split_idx = 0
        word_splits = [len(x) for x in seq_original]
        len_word_splits = len(word_splits)
        split_size_counter = 0

        om = offset_mapping[1:]
        bert_splits = []

        for omap in offset_mapping:
            split_size_counter += 1
            if omap[1] == word_splits[word_split_idx]:
                bert_splits.append(split_size_counter)
                split_size_counter = 0
                word_split_idx+=1
                if word_split_idx == len_word_splits:
                    break
        return bert_splits

    @staticmethod
    def bert_new_embedding(bert_encodings,split):
        return torch.stack([torch.mean(chunk,dim=0) for chunk in torch.split(bert_encodings[0],split)]).unsqueeze(0)

    def forward(self, bert_tokens, masks, text):
        bert_outs = []
        for toks, mask, txt in zip(bert_tokens, masks, text):
            bert_out = self.bert(toks, attention_mask=mask)[0]

            a = torch.sum(mask)
            seq_original = [w.lower() for w in txt.split(' ')]
            om = self.tokenizer.encode_plus(seq_original,  # the sentence to be encoded
                            add_special_tokens=True,  # Add [CLS] and [SEP]
                            pad_to_max_length=True,  # Add [PAD]s
                            is_split_into_words=True,
                            return_attention_mask = False,
                            return_offsets_mapping=True,
                            return_length=False)['offset_mapping'][1:]
            splits = self.parse_bert(seq_original, om)
            bert_out = bert_out[:,0:a,:]
            bert_out = bert_out[:,1:-1,:]
            bert_out = self.bert_new_embedding(bert_out,splits)
            bert_out = bert_out.permute(1, 0, 2)
            bert_outs.append(bert_out)

        token_splits = [
            len(t) for t in bert_outs
        ]

        return bert_outs, token_splits


class INNModel(pl.LightningModule):
    """INN model configuration.

    Parameters:
        vocab_dict (dict): The vocabulary for training, tokens to indices.
        word_embedding_dim (int): The size of the word embedding vectors.
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
        hidden_dim_bert,
        cell,
        freeze_bert_epoch,
        guided_training
    ):
        super().__init__()
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim_bert = hidden_dim_bert
        self.encoding_method = encoding_method
        self.guided_training = guided_training
        self.output_bert_hidden_states = output_bert_hidden_states

        if self.encoding_method == "bert":
            self.encoder = BERTEncoder(self.output_bert_hidden_states,freeze_bert_epoch)
            self.linear_in_dim = self.hidden_dim_bert
        else:
            self.encoder = FromScratchEncoder(vocab_dict, word_embedding_dim)
            self.linear_in_dim = 2 * self.word_embedding_dim

        self.element_embeddings = nn.Embedding(
            len(element_to_idx.keys()), self.linear_in_dim
        )

        self.attn_scores = nn.Linear(in_features=self.linear_in_dim, out_features=1)

        self.cell = cell

        self.output_linear = nn.Sequential(
            nn.Linear(self.linear_in_dim, 4 * self.word_embedding_dim),
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
            # self.attn_scores(torch.squeeze(encoding_out)),
            self.attn_scores(torch.cat(encoding_out).to(self.device)),
            token_splits
        )
        h_entities = []
        for sample in range(curr_batch_size):
            sample_entity_indices = [idx[idx >= 0] for idx in entity_indices[sample]]
            for idx in sample_entity_indices:
                sample_attn_scores = attn_scores[sample][idx.long()]
                sample_attn_weights = functional.softmax(sample_attn_scores, dim=0)
                encoding_vecs = encoding_out[sample][idx.long()]
                h_entity = torch.mul(sample_attn_weights, encoding_vecs).sum(axis=0)
                h_entities.append(h_entity.squeeze())

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
        text,
        entity_spans,
        element_names,
        T,
        S,
        L,
        labels,
        is_entity,
        epoch,
    ):
        encoding_out = None
        if self.encoding_method == "bert":
            encoding_out, token_splits = self.encoder(bert_tokens, mask,text,epoch)
        elif self.encoding_method == "from-scratch":
            encoding_out, token_splits = self.encoder(tokens)
        if encoding_out is None:
            raise ValueError("encoding did not occur check encoding_method")
        for i in range(len(encoding_out)):
            encoding_out[i] = encoding_out[i].to(self.device)
        curr_batch_size = len(entity_spans)

        # gets the hidden vector for each entity and stores them in H
        H = (
            torch.randn(T.shape[0], self.linear_in_dim)
            .detach()
            .to(self.device)
        )
        H = self.get_h_entities(
            entity_spans, encoding_out, token_splits, H, curr_batch_size, is_entity
        )

        C = (
            torch.zeros(T.shape[0], self.linear_in_dim)
            .detach()
            .to(self.device)
        )

        predictions = [
            PRED_TRUE if is_entity[i] == 1 else PRED_FALSE for i in range(0, T.shape[0])
        ]
        gold_predictions = [
            PRED_TRUE if v == 1 else PRED_FALSE for v in labels
        ]
        predictions = torch.stack(predictions).to(self.device)
        # predictions.requires_grad_()
        gold_predictions = torch.stack(gold_predictions).to(self.device)

        for layer in torch.unique(L):
            if layer > 0:
                # predictions = predictions.clone()
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
        train_distribution,
        learning_rate,
        freeze_bert_epoch,
        nll_positive_weight,
        guided_training
    ):
        super().__init__()
        self.encoding_method = encoding_method
        if self.encoding_method == "bert":
            encoding_dim = hidden_dim_bert
        else:
            encoding_dim = 2 * word_embedding_dim

        self.guided_training = guided_training

        self.cell = DAGLSTMCell(
            encoding_dim=encoding_dim,
            max_inputs=2,
        )
        self.inn = INNModel(
            vocab_dict=vocab_dict,
            element_to_idx=element_to_idx,
            encoding_method=encoding_method,
            output_bert_hidden_states=output_bert_hidden_states,
            hidden_dim_bert=hidden_dim_bert,
            word_embedding_dim=word_embedding_dim,
            cell=self.cell,
            freeze_bert_epoch=freeze_bert_epoch,
            guided_training=self.guided_training
        )
        self.nll_positive_weight = nll_positive_weight

        # loss criterion
        self.criterion = nn.NLLLoss(weight=torch.tensor([1,self.nll_positive_weight]))

        # step metrics
        self.accuracy = pl.metrics.Accuracy()
        self.f1 = pl.metrics.classification.F1()
        self.precision_score = pl.metrics.classification.Precision()
        self.recall =  pl.metrics.classification.Recall()
        self.confmat = pl.metrics.classification.ConfusionMatrix(num_classes=2)

        self.param_names = [p[0] for p in self.inn.named_parameters()]
        self.training_candidates = 0 # counter for how many candidates the model has seen
        self.training_samples =  0
        self.lr = learning_rate

        self.train_distribution = train_distribution

    def forward(self, batch_sample):
        predictions = self.inn(*self.expand_batch(batch_sample))
        return predictions

    def convert_predictions(self,predicted_probs):
        """ takes predicted_probs (softmax output) and returns
        predicted_probs with logarithm applied and binary labels """
        log_predicted_probs = torch.log(predicted_probs)
        predicted_labels = predicted_probs[:, 1] > 0.5

        return log_predicted_probs, predicted_labels

    def _get_naive_all_neg_predicted_probs(self,labels):
        naive_preds = [
            PRED_FALSE for _ in labels
        ]
        naive_preds = torch.stack(naive_preds).to(self.device)
        return naive_preds

    def _get_naive_random_predicted_probs(self, num_labels, flag):
        random_nums = torch.rand(num_labels)
        percent_ones = self.train_distribution[flag].item()

        naive_preds = [
            PRED_TRUE if random_nums[i].item() < percent_ones else PRED_FALSE for i in range(num_labels)
        ]
        naive_preds = torch.stack(naive_preds).to(self.device)

        return naive_preds

    def _calculate_metrics(self,predicted_probs,log_predicted_probs,predicted_labels,true_labels,batch_size,prefix,avg_strategy):
        """
        calculate  various performance metrics based on predicted probabilities and true labels
        """

        metrics = {}
        metrics["acc"] = self.accuracy(predicted_probs, true_labels)
        metrics["f1"] = self.f1(predicted_labels, true_labels)
        confmat = self.confmat(predicted_labels, true_labels)
        tn, fp, fn, tp = confmat.flatten().tolist()
        metrics["confmat"] = confmat
        metrics["true_pos"] = tp
        metrics["false_pos"] = fp
        metrics["false_neg"] = fn
        metrics["true_neg"] = tn
        metrics["precision"] = self.precision_score(predicted_labels, true_labels)
        metrics["recall"] = self.recall(predicted_labels, true_labels)
        metrics["num_candidates"] = len(true_labels)
        metrics["predicted_pos"] = torch.sum(predicted_labels)
        metrics["batch_size"] = batch_size
        metrics["epoch"] = self.current_epoch
        metrics["training_candidates"] = self.training_candidates
        metrics["training_samples"] = self.training_samples


        return {f'{prefix}_{k}_{avg_strategy}_avg': torch.tensor(v).float().cpu() for k, v in metrics.items()}

    def _calculate_step_metrics_and_loss(self,predicted_probs,true_labels,batch_size,prefix,avg_strategy):
        """
        Calculates performance metrics for model predictions as well as several naive strategies
        """
        naive_neg_predicted_probs = self._get_naive_all_neg_predicted_probs(true_labels)
        naive_rand_predicted_probs1 = self._get_naive_random_predicted_probs(true_labels.numel(), 0)
        naive_rand_predicted_probs2 = self._get_naive_random_predicted_probs(true_labels.numel(), 1)

        log_predicted_probs, predicted_labels = self.convert_predictions(predicted_probs)
        log_naive_neg_predicted_probs, naive_neg_predicted_labels = self.convert_predictions(naive_neg_predicted_probs)
        log_naive_rand_predicted_probs1, naive_rand_predicted_labels1 = self.convert_predictions(naive_rand_predicted_probs1)
        log_naive_rand_predicted_probs2, naive_rand_predicted_labels2 = self.convert_predictions(naive_rand_predicted_probs2)

        metrics = self._calculate_metrics(predicted_probs,log_predicted_probs,
                                               predicted_labels,true_labels,batch_size,prefix=prefix,avg_strategy=avg_strategy)

        naive_neg_metrics = self._calculate_metrics(naive_neg_predicted_probs,log_naive_neg_predicted_probs,
                                                         naive_neg_predicted_labels,true_labels,
                                                         batch_size, prefix=f"naive_all_neg/{prefix}",avg_strategy=avg_strategy)

        naive_rand_metrics1 = self._calculate_metrics(naive_rand_predicted_probs1, log_naive_rand_predicted_probs1,
                                                          naive_rand_predicted_labels1, true_labels,
                                                          batch_size, prefix=f"naive_random1/{prefix}",avg_strategy=avg_strategy)

        naive_rand_metrics2 = self._calculate_metrics(naive_rand_predicted_probs2, log_naive_rand_predicted_probs2,
                                                          naive_rand_predicted_labels2, true_labels,
                                                          batch_size, prefix=f"naive_random2/{prefix}",avg_strategy=avg_strategy)

        loss = self.criterion(log_predicted_probs, true_labels)
        metrics["train/loss"] = loss.cpu()

        return loss, metrics, naive_neg_metrics, naive_rand_metrics1, naive_rand_metrics2


    def training_step(self, batch_sample, batch_idx):
        predicted_probs = self.inn(*self.expand_batch(batch_sample))

        true_labels = batch_sample["labels"]
        self.training_candidates += len(true_labels)
        batch_size = len(batch_sample["entity_spans"])
        self.training_samples += batch_size
        loss, metrics, naive_neg_metrics, \
        naive_rand_metrics1, naive_rand_metrics2 = self._calculate_step_metrics_and_loss(predicted_probs,
                                                                                         true_labels,
                                                                                         batch_size,
                                                                                         prefix='train',avg_strategy='batch')
        self.logger.experiment.log(metrics)
        self.logger.experiment.log(naive_neg_metrics)
        self.logger.experiment.log(naive_rand_metrics1)
        self.logger.experiment.log(naive_rand_metrics2)

        opt = self.optimizers()
        if metrics["train_predicted_pos_batch_avg"] > len(batch_sample["entity_spans"]):
            self.manual_backward(loss, opt)
            opt.step()

    def expand_batch(self, batch_sample):
        return (
            batch_sample["from_scratch_tokens"],
            batch_sample["bert_tokens"],
            batch_sample["mask"],
            batch_sample["text"],
            batch_sample["entity_spans"],
            batch_sample["element_names"],
            batch_sample["T"],
            batch_sample["S"],
            batch_sample["L"],
            batch_sample["labels"],
            batch_sample["is_entity"],
            self.current_epoch,
        )

    def validation_step(self, batch_sample, batch_idx):
        predicted_probs = self.inn(*self.expand_batch(batch_sample))
        return predicted_probs, batch_sample

    def test_step(self, batch_sample, batch_idx):
        predicted_probs = self.inn(*self.expand_batch(batch_sample))
        return predicted_probs, batch_sample

    def log_averages(self,m):
        for metric in m[0].keys():
            avg_val = torch.stack([x[metric] for x in m]).mean()
            self.logger.experiment.log({metric: avg_val},commit=False)

    def _log_epoch_end_performance_sample_avg(self,step_outputs,prefix):
        metrics = []
        naive_neg_metrics = []
        naive_rand_metrics1 = []
        naive_rand_metrics2 = []

        for batch_probs, batch_sample in step_outputs:
            batch_labels = batch_sample['labels']
            loss, batch_metrics, batch_naive_neg_metrics, batch_naive_rand_metrics1, batch_naive_rand_metrics2 = self._calculate_step_metrics_and_loss(batch_probs,batch_labels,batch_size=len(step_outputs),prefix=prefix,avg_strategy='sample') #depends on batch_size=1 for this to actually be "sample" avg
            metrics.append(batch_metrics)
            naive_neg_metrics.append(batch_naive_neg_metrics)
            naive_rand_metrics1.append(batch_naive_rand_metrics1)
            naive_rand_metrics2.append(batch_naive_rand_metrics2)

        for metrs in [metrics, naive_neg_metrics, naive_rand_metrics1, naive_rand_metrics2]:
            self.log_averages(metrs)

    def _log_epoch_end_performance_full_set_avg(self,step_outputs,prefix):
        predicted_probs = torch.cat([o[0] for o in step_outputs])
        batch_labels = torch.cat([o[1]["labels"] for o in step_outputs])
        loss, metrics, naive_neg_metrics, \
        naive_rand_metrics1, naive_rand_metrics2 = self._calculate_step_metrics_and_loss(predicted_probs,
                                                                                         batch_labels,
                                                                                         batch_size=len(step_outputs),
                                                                                         prefix=prefix,avg_strategy='full_set')
        metrics[f"{prefix}_loss"] = loss.cpu()
        for metrs in [metrics, naive_neg_metrics, naive_rand_metrics1, naive_rand_metrics2]:
            self.logger.experiment.log(metrs, commit=False)

    def validation_epoch_end(self, val_step_outputs):
        self._log_epoch_end_performance_sample_avg(val_step_outputs,prefix='val')
        self._log_epoch_end_performance_full_set_avg(val_step_outputs,prefix='val')

    def test_epoch_end(self, test_step_outputs):
        self._log_epoch_end_performance_sample_avg(test_step_outputs,prefix='test')
        self._log_epoch_end_performance_full_set_avg(test_step_outputs,prefix='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=self.lr)
        return optimizer
