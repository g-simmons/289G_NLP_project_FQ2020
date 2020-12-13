import torch
from torch import nn
from torch.nn import functional as functional
import pytorch_lightning as pl

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
)


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

        self.W_ioc_hat = nn.Linear(relation_embedding_dim, 3 * hidden_dim, bias=False)
        self.W_fs = nn.Linear(relation_embedding_dim, hidden_dim, bias=False)

        self.U_ioc_hat = nn.Linear(max_inputs * hidden_dim, 3 * hidden_dim, bias=False)
        self.U_fs = nn.Linear(max_inputs * hidden_dim, hidden_dim, bias=False)

        self.b_ioc_hat = nn.Parameter(torch.zeros(3 * hidden_dim))
        self.b_fs = nn.Parameter(torch.zeros(hidden_dim))

    def init_cell_state(self):
        return torch.zeros(1, self.hidden_dim).clone().detach().requires_grad_(True)

    def forward(self, H, C, element_embeddings, S):
        v = torch.stack([torch.flatten(H[idx]) for idx in S])
        cell_states = torch.stack([torch.flatten(C[idx]) for idx in S])
        print(v.shape)
        print(element_embeddings.shape)
        print(cell_states.shape)

        ioc_hat = self.W_ioc_hat(element_embeddings)
        ioc_hat += self.U_ioc_hat(v)
        ioc_hat += self.b_ioc_hat

        i, o, c_hat = torch.chunk(ioc_hat, 3, dim=1)
        i, o, c_hat = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(c_hat)

        ebc = element_embeddings.repeat(1, self.max_inputs).reshape(
            -1, element_embeddings.shape[1]
        )
        fj = self.W_fs(ebc)


        vbc = v.repeat(1, self.max_inputs).reshape(-1, v.shape[1])
        fj += self.U_fs(vbc)
        fj += self.b_fs

        print(fj.shape)
        print(cell_states.shape)

        fj = torch.sigmoid(fj)

        fjcj = torch.mul(fj, cell_states)
        fj_mul_css = torch.stack([torch.sum(fjcj[idx], dim=0) for idx in S])

        c = torch.mul(i, c_hat) + fj_mul_css

        h = torch.mul(torch.tanh(c), o)

        return h, c
