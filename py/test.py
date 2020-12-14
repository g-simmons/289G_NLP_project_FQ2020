import torch

from config import *
from daglstmcell import DAGLSTMCell


class TestDAGLSTM:
    def test_dag_lstm_forward(self):
        MAX_INPUTS = 2
        HIDDEN_DIM = 256
        RELATION_EMBEDDING_DIM = HIDDEN_DIM

        cell = DAGLSTMCell(
            hidden_dim= HIDDEN_DIM,
            relation_embedding_dim=RELATION_EMBEDDING_DIM,
            max_inputs=MAX_INPUTS,
            hidden_state_clamp_val=HIDDEN_STATE_CLAMP_VAL,
            cell_state_clamp_val=CELL_STATE_CLAMP_VAL,
        )

        H = torch.randn(10 * MAX_INPUTS, HIDDEN_DIM)
        C = (
            torch.zeros(H.shape[0], HIDDEN_DIM)
            .clone()
            .detach()
            .requires_grad_(True)
        )
        S = torch.arange(0, 10 * MAX_INPUTS).reshape(10, MAX_INPUTS)
        E = torch.randn(10, RELATION_EMBEDDING_DIM)

        h, c = cell.forward(H, C, E, S)
        assert tuple(h.shape) == (10, HIDDEN_DIM)
        assert tuple(c.shape) == (10, HIDDEN_DIM)


class TestTrain():
    def test_collate_func(self):
        sample_1 = {
            "tokens": torch.tensor([3,4,5]),
            "entity_spans": torch.tensor([[0],[1,2]]),
            "element_names": torch.tensor([51,52]),
            "T": torch.tensor([0,1,2,3,4,5]),
            "L": torch.tensor([1,1,0,1,0]),
            "layers": torch.tensor([0,0,1,1,1]),
            "is_entity": torch.tensor([1,1,0,0,0])
        }
        sample_2 = {
            "tokens": torch.tensor([3,4,5]),
            "entity_spans": torch.tensor([[0],[1,2]]),
            "element_names": torch.tensor([51,52]),
            "T": torch.tensor([0,1,2,3,4,5]),
            "L": torch.tensor([1,1,0,1,0]),
            "layers": torch.tensor([0,0,1,1,1]),
            "is_entity": torch.tensor([1,1,0,0,0])
        }

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
