import pytest
import torch

from config import *
from daglstmcell import DAGLSTMCell
from INN import INNModel
from train import collate_func


class TestDAGLSTM:
    def test_dag_lstm_forward(self):
        MAX_INPUTS = 2

        HIDDEN_DIM = 512
        RELATION_EMBEDDING_DIM = HIDDEN_DIM // 2

        cell = DAGLSTMCell(
            hidden_dim=HIDDEN_DIM,
            relation_embedding_dim=RELATION_EMBEDDING_DIM,
            max_inputs=MAX_INPUTS,
            hidden_state_clamp_val=HIDDEN_STATE_CLAMP_VAL,
            cell_state_clamp_val=CELL_STATE_CLAMP_VAL,
        )

        H = torch.randn(10 * MAX_INPUTS, HIDDEN_DIM)
        S = torch.arange(0, 10 * MAX_INPUTS).reshape(10, MAX_INPUTS)
        E = torch.randn(10, RELATION_EMBEDDING_DIM)
        c = cell.init_cell_state()
        C = torch.cat([c for _ in H])

        h, c = cell.forward(H, C, E, S)
        assert tuple(h.shape) == (10, HIDDEN_DIM)
        assert tuple(c.shape) == (10, HIDDEN_DIM)


# class TestINNModel():
#     def test_forward():
#         batch_sample = []
#         batch_sample["tokens"] = torch.randn(50)
#         batch_sample["entity_spans"] = torch.
#         batch_sample["element_names"] =
#         batch_sample["T"],
#         batch_sample["S"],
#         batch_sample["entity_spans_pre-padded_size"],
#         batch_sample["tokens_pre-padded_size"],
#     def test_collate_fn(self):
#         sample_1 = {}
#         sample_2 = {}
