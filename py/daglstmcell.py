import pytorch_lightning as pl
import torch
from torch import nn
from config import *


class DAGLSTMCell(pl.LightningModule):
    # credit to https://github.com/dmlc/dgl/tree/master/examples/pytorch/tree_lstm for the
    # original tree-lstm implementation, modified here
    def __init__(
        self,
        hidden_dim: int,
        relation_embedding_dim: int,
        max_inputs: int,
        hidden_state_clamp_val: float,
        cell_state_clamp_val: float,
    ):
        super(DAGLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.relation_embedding_dim = relation_embedding_dim
        self.max_inputs = max_inputs
        self.hidden_state_clamp_val = hidden_state_clamp_val
        self.cell_state_clamp_val = cell_state_clamp_val

        ioc_out_dim = 3 * hidden_dim
        self.W_ioc_hat = nn.Linear(relation_embedding_dim, ioc_out_dim, bias=False)
        self.U_ioc_hat = nn.Linear(max_inputs * hidden_dim, ioc_out_dim, bias=False)
        self.b_ioc_hat = nn.Parameter(torch.zeros(ioc_out_dim))

        f_out_dim = 2 * hidden_dim
        self.W_fs = nn.Linear(relation_embedding_dim, f_out_dim, bias=False)
        self.U_fs = nn.Linear(max_inputs * hidden_dim, f_out_dim, bias=False)
        self.b_fs = nn.Parameter(torch.zeros(f_out_dim))

    def init_cell_state(self):
        return torch.zeros(1, self.hidden_dim).clone().detach().requires_grad_(True)

    def _validate_inputs(self, H, C, element_embeddings, S):
        n_args = S.shape[0] * 2
        n_entities = element_embeddings.shape[0]
        assert n_args == 2 * n_entities
        assert element_embeddings.shape[1] == self.relation_embedding_dim
        assert S.shape[0] == n_entities
        assert S.shape[1] == 2

        return H, C, element_embeddings, S, n_entities, n_args

    def _val_fj_cell_states(self, fj, cell_states, n_args):
        assert cell_states.shape[0] == n_args
        assert fj.shape[0] == n_args
        assert cell_states.shape[1] == self.hidden_dim
        assert fj.shape[1] == self.hidden_dim

    def forward(self, hidden_vectors, cell_states, element_embeddings, S):
        # if VAL_DIMS:
        #     H, C, element_embeddings, S, n_entities, n_args = self._validate_inputs(
        #         H, C, element_embeddings, S
        #     )

        v = hidden_vectors

        ioc_hat = self.W_ioc_hat(element_embeddings)
        ioc_hat += self.U_ioc_hat(v)
        ioc_hat += self.b_ioc_hat

        i, o, c_hat = torch.chunk(ioc_hat, 3, dim=1)
        i, o, c_hat = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(c_hat)

        fj = torch.sigmoid(self.W_fs(element_embeddings) + self.U_fs(v) + self.b_fs)

        fj = fj.reshape(-1, fj.shape[1] // 2)

        cell_states = cell_states.reshape(-1, cell_states.shape[1] // 2)
        if VAL_DIMS:
            self._val_fj_cell_states(fj, cell_states, n_args)

        fjcj = torch.mul(fj, cell_states)
        fj_mul_css = torch.stack(
            [torch.sum(fjcj_split, dim=0) for fjcj_split in fjcj.split(2)]
        )

        c = torch.mul(i, c_hat) + fj_mul_css

        h = torch.mul(torch.tanh(c), o)

        return h, c
