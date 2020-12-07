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
    HIDDEN_STATE_CLAMP_VAL
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
            attn_scores = torch.index_select(attn_scores_out, dim=0, index=idx)
            attn_weights = functional.softmax(attn_scores, dim=0)
            h_entity = torch.matmul(attn_weights, blstm_out[idx])
            h_entity = h_entity.sum(axis=0)
            H_new[i] = h_entity
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

        blstm_out, _ = self.blstm(
            embedded_sentence.view(embedded_sentence.shape[0], 1, -1)
        )

        H = self.get_h_entities(entity_spans, blstm_out, H)

        predictions = []
        for i in range(0,len(entity_spans)):
            predictions.append(torch.tensor([0.001, 0.999]).to(self.device))

        c = self.cell.init_cell_state()
        c = c.to(self.device)

        for argset, target_idx, element_name in zip(S, T, element_names):
            if target_idx >= len(entity_spans):
                args_idx = argset[argset > -1]
                preds = torch.stack(predictions)
                if torch.all(preds[args_idx, 1] > 0.5):
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
                    predictions.append(torch.tensor([0.999, 0.001]).to(self.device))
        predictions = torch.stack(predictions)

        return predictions

class INNModelLightning(pl.LightningModule):
  def __init__(self, vocab_dict, element_to_idx, word_embedding_dim, relation_embedding_dim, hidden_dim, cell_state_clamp_val, hidden_state_clamp_val):
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
    self.tb = SummaryWriter()

  def forward(self, x):
    tokens, entity_spans, element_names, A, T, S, labels = self.expand_batch(x)
    predictions = self.inn(tokens, entity_spans, element_names, A, T, S)
    return predictions

  def expand_batch(self, batch):
    tokens = sample['tokens']
    entity_spans = sample['entity_spans']
    element_names = sample['element_names']
    A = sample['A']
    T = sample['T']
    S = sample['S']
    labels = sample['labels']
    return tokens, entity_spans, element_names, A, T, S, labels

  def training_step(self, batch, batch_idx):
    opt = self.optimizers()
    tokens, entity_spans, element_names, A, T, S, labels = self.expand_batch(batch)
    raw_predictions = self.inn(tokens, entity_spans, element_names, A, T, S)
    predictions = torch.log(raw_predictions)
    loss = self.criterion(predictions, labels)
    if len(predictions) > len(entity_spans):
        self.manual_backward(loss, opt)
        self.manual_optimizer_step(opt)
        self.log('loss',loss)

  def validation_step(self, batch, batch_idx):
    tokens, entity_spans, element_names, A, T, S, labels = self.expand_batch(batch)
    raw_predictions = self.inn(tokens, entity_spans, element_names, A, T, S)
    predictions = torch.log(raw_predictions)
    loss = self.criterion(predictions, labels)
    return loss

  def process_sample(self, sample, inverse_schema):
    element_names = sample["element_names"]
    j = len(element_names)
    element_indices = torch.arange(j)

    S_temp = [
        nn.functional.pad(e, pad=(0, 2 - len(e)), mode="constant", value=-1)
        for e in list(element_indices.chunk(j))
    ]
    T_temp = element_indices.tolist()

    a = 1  # TODO: only handling single sentences for now
    A_temp = [a for _ in element_indices]
    labels_temp = [1 for _ in element_indices]

    max_layers = MAX_LAYERS

    for _ in range(max_layers):
        ttt = torch.tensor(T_temp)
        for c in torch.combinations(ttt):  # TODO single-argument relations?
            e_names = torch.tensor(element_names)[c]
            key = tuple(sorted(e.item() for e in e_names))
            if key in inverse_schema.keys():
                for predicate in inverse_schema[key].keys():
                    S_temp.append(c)
                    T_temp.append(j)
                    A_temp.append(a)
                    element_names = torch.cat((element_names, torch.tensor([[predicate]]).to(self.device)))
                    L = 0  # default label is false
                    for i, g in enumerate(sample["relation_graphs"]):
                        for n in g.nodes():
                            child_idx = get_child_indices(g, node_idx=n)
                            child_idx = torch.tensor(
                                [
                                    sample["node_idx_to_element_idxs"][i][idx]
                                    for idx in child_idx
                                ]
                            )
                            if child_idx.shape == c.shape:
                                # TODO ordering
                                if (
                                    child_idx.tolist() == c.tolist()
                                    and element_names[j] == g.ndata["element_names"][n]
                                ):  # check if children match and the predicate type is correct
                                    sample["node_idx_to_element_idxs"][i][n.item()] = j
                                    L = 1  # this label is true because we found this candidate in the gold standard relation graphs
                    labels_temp.append(L)
                    j += 1

    sample["labels"] = torch.tensor(labels_temp, dtype=torch.long).to(self.device)
    sample["A"] = torch.tensor(A_temp).to(self.device)
    sample["T"] = torch.tensor(T_temp).to(self.device)
    sample["S"] = torch.stack(S_temp).to(self.device)
    sample["element_names"] = torch.tensor(element_names).to(self.device)
    sample["H"] = torch.randn(sample["A"].shape[0], 2 * HIDDEN_DIM).to(self.device)

    return sample

  def configure_optimizers(self):
    optimizer = torch.optim.Adadelta(
        self.parameters(), lr=1.0
    )
    return optimizer