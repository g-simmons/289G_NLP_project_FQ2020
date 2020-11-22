import torch
from torch import nn
from torch.nn import functional as functional

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
)


class DAGLSTMCell(nn.Module):
    # credit to https://github.com/dmlc/dgl/tree/master/examples/pytorch/tree_lstm for the
    # original tree-lstm implementation, modified here
    def __init__(
        self,
        hidden_dim: int,
        relation_embedding_dim: int,
        max_inputs: int,
    ):
        super(DAGLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.relation_embedding_dim = relation_embedding_dim
        self.max_inputs = max_inputs

        self.W_ioc_hat = nn.Linear(relation_embedding_dim, 3 * hidden_dim, bias=False)
        self.W_fs = nn.ModuleList(
            [
                nn.Linear(relation_embedding_dim, hidden_dim, bias=False)
                for j in range(max_inputs)
            ]
        )

        self.U_ioc_hat = nn.Linear(
            max_inputs * hidden_dim, 3 * hidden_dim, bias=False
        )  # can pad with zeros to have dynamic
        self.U_fs = nn.ModuleList(
            [
                nn.Linear(max_inputs * hidden_dim, hidden_dim, bias=False)
                for j in range(max_inputs)
            ]
        )

        self.b_ioc_hat = nn.Parameter(torch.zeros(3 * hidden_dim))
        self.b_fs = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_dim)) for j in range(max_inputs)]
        )

    def init_cell_state(self):
        return torch.zeros(1, self.hidden_dim).clone().detach().requires_grad_(True)

    def forward(self, hjs, cjs, e):
        v = torch.flatten(hjs)

        ioc_hat = self.W_ioc_hat(e)

        ioc_hat += self.U_ioc_hat(v)
        ioc_hat += self.b_ioc_hat
        ioc_hat = torch.sigmoid(ioc_hat)
        i, o, c_hat = torch.chunk(ioc_hat, 3)
        i, o, c_hat = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(c_hat)

        fj_mul_cs = torch.zeros(1, self.hidden_dim)
        for j in range(len(hjs)):
            fj = torch.sigmoid(self.W_fs[j](e) + self.U_fs[j](v) + self.b_fs[j])
            fj_mul_cs += torch.mul(fj, cjs[j])

        c = torch.mul(i, c_hat) + torch.sum(fj_mul_cs)

        h = torch.mul(torch.tanh(c), o)

        return h, c
