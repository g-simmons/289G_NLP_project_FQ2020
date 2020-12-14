import pytorch_lightning as pl
import torch
from torch import nn

from config import *


class DAGLSTMCell(pl.LightningModule):
    # credit to https://github.com/dmlc/dgl/tree/master/examples/pytorch/tree_lstm for the
    # original tree-lstm implementation, modified here
    def __init__(
        self,
        blstm_out_dim: int,
        max_inputs: int,
        hidden_state_clamp_val: float,
        cell_state_clamp_val: float,
    ):
        super(DAGLSTMCell, self).__init__()
        self.hidden_dim = blstm_out_dim
        self.relation_embedding_dim = self.hidden_dim
        self.max_inputs = max_inputs
        self.hidden_state_clamp_val = hidden_state_clamp_val
        self.cell_state_clamp_val = cell_state_clamp_val

        ioc_out_dim = 3 * self.hidden_dim
        self.W_ioc_hat = nn.Linear(self.relation_embedding_dim, ioc_out_dim, bias=False)
        self.U_ioc_hat = nn.Linear(
            max_inputs * self.hidden_dim, ioc_out_dim, bias=False
        )
        self.b_ioc_hat = nn.Parameter(torch.zeros(ioc_out_dim))

        f_out_dim = 2 * self.hidden_dim
        self.W_fs = nn.Linear(self.relation_embedding_dim, f_out_dim, bias=False)
        self.U_fs = nn.Linear(max_inputs * self.hidden_dim, f_out_dim, bias=False)
        self.b_fs = nn.Parameter(torch.zeros(f_out_dim))

    def init_cell_state(self):
        return torch.zeros(1, self.hidden_dim).clone().detach().requires_grad_(True)

    def _validate_inputs(self, hidden_vectors, cell_states, element_embeddings, S):
        if not element_embeddings.shape[0] == S.shape[0]:
            raise ValueError("element embeddings and S should have same shape[0]")
        if not hidden_vectors.shape[0] == cell_states.shape[0]:
            raise ValueError("hidden_vectors and cell_states should have same shape[0]")

        n_entities = element_embeddings.shape[0]
        n_argsets = hidden_vectors.shape[0]

        if not n_argsets == n_entities:
            raise ValueError("shape[0] for hidden_vectors and cell_states should be twice shape[0] for element embeddings and S")

        if not S.shape[1] == 2:
            raise ValueError("S.shape[1] should be 2")

        if not hidden_vectors.shape[1] == cell_states.shape[1] == self.hidden_dim * 2:
            raise ValueError(f"hidden_vectors, cell_states should have shape[1] == self.hidden_dim, but have shapes {hidden_vectors.shape}, {cell_states.shape}")

        return hidden_vectors, cell_states, element_embeddings, S, n_entities, n_argsets

    def _val_fj_cell_states(self, fj, cell_states, n_argsets):
        assert cell_states.shape[0] == n_argsets
        assert fj.shape[0] == n_argsets
        assert cell_states.shape[1] == self.hidden_dim
        assert fj.shape[1] == self.hidden_dim

    def forward(self, hidden_vectors, cell_states, element_embeddings, S):
        hidden_vectors, cell_states, element_embeddings, S, n_entities, n_argsets = self._validate_inputs(hidden_vectors, cell_states, element_embeddings, S)
        v = hidden_vectors

        ioc_hat = self.W_ioc_hat(element_embeddings)
        ioc_hat += self.U_ioc_hat(v)
        ioc_hat += self.b_ioc_hat

        i, o, c_hat = torch.chunk(ioc_hat, 3, dim=1)
        i, o, c_hat = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(c_hat)

        f = torch.sigmoid(self.W_fs(element_embeddings) + self.U_fs(v) + self.b_fs)

        cell_states = cell_states
        fcj = torch.mul(f, cell_states)

        fj_mul_css = torch.sum(torch.stack([fcj_x for fcj_x in torch.split(fcj, self.hidden_dim, dim=1)]),dim=0)

        c = torch.mul(i, c_hat) + fj_mul_css

        h = torch.mul(torch.tanh(c), o)

        return h, c
