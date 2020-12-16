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

    def parse_bert(self, seq_original, seq_bert):
        """
        """

        def remove_leading_pounds(token):
            """
            """
            token_new = ''
            for c in token:
                if c != '#':
                    token_new += c
            return token_new

        # Iterate over the original sequence and detect splitted tokens.
        mapped_indices_list = []
        j = 0
        for i in range(len(seq_original)):
            if seq_original[i] == seq_bert[j]:  # Not splitted.
                j += 1
                continue
            else:  # Detect splitted tokens.
                start = 0
                token_splitted = seq_original[i]
                token_mapping = remove_leading_pounds(seq_bert[j])
                mapped_indices = []
                while token_mapping == \
                        token_splitted[start : start + len(token_mapping)]:
                    mapped_indices.append(j)
                    start += len(token_mapping)
                    j += 1
                    token_mapping = remove_leading_pounds(seq_bert[j])
                mapped_indices_list.append((i, mapped_indices))
        return mapped_indices_list

    def Bert_New_Embedding(self,bert,split,shape):
        '''
        '''
        j=0
        k=0
        bert_new=torch.zeros([1,shape,768])
        if len(split) != 0:
            for i in range(len(split)):
                while k< split[i][0]:
                    bert_new[:,k,:]=bert[:,j,:]
                    j += 1
                    k += 1
                for p in range(len(split[i][1])):
                    j += 1

                bert_new[:,k,:] = torch.sum(bert[:,split[i][1],:],dim = 1)/len(split[i][1])
                k +=1
            while k < shape:
                bert_new[:,k,:]=bert[:,j,:]
                j += 1
                k += 1
        else:
            bert_new = bert
            
        return bert_new

    def forward(self, bert_tokens, masks,text):
        tokens = bert_tokens

        bert_outs = []
        for toks, mask, txt in zip(tokens, masks,text):
            bert_out = self.bert(toks, attention_mask=mask)[0]

            a = torch.sum(mask)
            seq_original = [w.lower() for w in txt.split(' ')]
            seq_bert = self.tokenizer.tokenize(txt)
            splits = self.parse_bert(seq_original, seq_bert)
            bert_out = bert_out[:,0:a,:]
            bert_out = bert_out[:,1:-1,:]
            bert_out = self.Bert_New_Embedding(bert_out,splits,len(seq_original))
            bert_out = bert_out.permute(1, 0, 2)
            bert_outs.append(bert_out)
            
        token_splits = [
            len(t) for t in bert_outs
        ]  # should be tokens or bert_tokens? check len matches later on

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
        cell_state_clamp_val,
        hidden_state_clamp_val,
    ):
        super().__init__()
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim_bert = hidden_dim_bert
        self.encoding_method = encoding_method
        self.output_bert_hidden_states = output_bert_hidden_states

        if self.encoding_method == "bert":
            self.encoder = BERTEncoder(self.output_bert_hidden_states)
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
        is_entity,
    ):
        encoding_out = None
        if self.encoding_method == "bert":
            encoding_out, token_splits = self.encoder(bert_tokens, mask,text)
        elif self.encoding_method == "from-scratch":
            encoding_out, token_splits = self.encoder(tokens)
        if encoding_out is None:
            raise ValueError("encoding did not occur check encoding_method")
        for i in range(len(encoding_out)):
            encoding_out[i] = encoding_out[i].to(self.device)
        print("encoding_out:", encoding_out[0].device)
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
            cell=self.cell,
            cell_state_clamp_val=cell_state_clamp_val,
            hidden_state_clamp_val=hidden_state_clamp_val,
        )

        # loss criterion
        self.criterion = nn.NLLLoss()

        # step metrics
        self.accuracy = pl.metrics.Accuracy()
        self.f1 = pl.metrics.classification.F1()
        self.confmat = pl.metrics.classification.ConfusionMatrix(num_classes=2)

        self.param_names = [p[0] for p in self.inn.named_parameters()]

    def forward(self, batch_sample):
        predictions = self.inn(*self.expand_batch(batch_sample))
        return predictions

    def convert_predictions(self,predicted_probs):
        """ takes predicted_probs (softmax output) and returns
        predicted_probs with logarithm applied and binary labels """
        log_predicted_probs = torch.log(predicted_probs)
        predicted_labels = predicted_probs[:, 1] > 0.5

        return log_predicted_probs, predicted_labels

    def _get_naive_predicted_probs(self,labels):
        naive_preds = [
            PRED_FALSE for _ in labels
        ]
        naive_preds = torch.stack(naive_preds).to(self.device)
        return naive_preds

    def _calculate_step_metrics(self,predicted_probs,log_predicted_probs,predicted_labels,batch_sample,batch_size,prefix):

        true_labels = batch_sample["labels"]

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
        metrics["num_candidates"] = len(batch_sample["labels"])
        metrics["predicted_pos"] = torch.sum(predicted_labels)
        metrics["batch_size"] = batch_size

        return {f'{prefix}_{k}': torch.tensor(v).float().cpu() for k, v in metrics.items()}

    def _calculate_step_metrics_and_loss(self,predicted_probs,batch_sample,prefix):
        batch_size = len(batch_sample["entity_spans"])
        naive_predicted_probs = self._get_naive_predicted_probs(batch_sample['labels'])

        log_predicted_probs, predicted_labels = self.convert_predictions(predicted_probs)
        log_naive_predicted_probs, naive_predicted_labels = self.convert_predictions(naive_predicted_probs)

        metrics = self._calculate_step_metrics(predicted_probs,log_predicted_probs,predicted_labels,batch_sample,batch_size,prefix=prefix)
        naive_metrics = self._calculate_step_metrics(naive_predicted_probs,log_naive_predicted_probs,naive_predicted_labels,batch_sample,batch_size,prefix=f"naive_{prefix}")

        loss = self.criterion(log_predicted_probs, batch_sample["labels"])
        metrics["train_loss"] = loss.cpu()

        return loss, metrics, naive_metrics


    def training_step(self, batch_sample, batch_idx):
        print("training_step:", self.device)
        predicted_probs = self.inn(*self.expand_batch(batch_sample))

        loss, metrics, naive_metrics = self._calculate_step_metrics_and_loss(predicted_probs,batch_sample,prefix='train')
        self.logger.experiment.log(metrics)
        self.logger.experiment.log(naive_metrics)

        opt = self.optimizers()
        if metrics["train_predicted_pos"] > len(batch_sample["entity_spans"]):
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
            batch_sample["is_entity"],
        )

    def validation_step(self, batch_sample, batch_idx):
        predicted_probs = self.inn(*self.expand_batch(batch_sample))
        loss, metrics, naive_metrics = self._calculate_step_metrics_and_loss(predicted_probs,batch_sample,prefix='val')
        return metrics, naive_metrics

    def validation_epoch_end(self, val_step_outputs):
        metrics = [o[0] for o in val_step_outputs]
        naive_metrics = [o[1] for o in val_step_outputs]

        def log_averages(m):
            for metric in m[0].keys():
                avg_val = torch.stack([x[metric] for x in m]).mean()
                self.logger.experiment.log({metric: avg_val},commit=False)

        log_averages(metrics)
        log_averages(naive_metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=LEARNING_RATE)
        return optimizer
