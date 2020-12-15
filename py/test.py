import pytest
import torch

from config import *
from daglstmcell import DAGLSTMCell
from train import collate_func


class TestDAGLSTM:
    def test_dag_lstm_forward(self):
        MAX_INPUTS = 2
        HIDDEN_DIM = 512

        cell = DAGLSTMCell(
            blstm_out_dim=HIDDEN_DIM,
            max_inputs=MAX_INPUTS,
            hidden_state_clamp_val=HIDDEN_STATE_CLAMP_VAL,
            cell_state_clamp_val=CELL_STATE_CLAMP_VAL,
        )

        H = torch.randn(10 * MAX_INPUTS, HIDDEN_DIM).reshape(10, -1)
        C = torch.zeros_like(H)
        S = torch.arange(0, 10 * MAX_INPUTS).reshape(10, MAX_INPUTS)
        E = torch.randn(10, HIDDEN_DIM)

        h, c = cell.forward(H, C, E, S)
        assert tuple(h.shape) == (10, HIDDEN_DIM)
        assert tuple(c.shape) == (10, HIDDEN_DIM)


class TestTrain:
    @pytest.fixture
    def sample_1(self):
        sample_1 = {
            "tokens": torch.tensor([3, 4, 5, 6, 7, 8]),
            "entity_spans": torch.tensor([[0, -1], [1, 2]]),
            "element_names": torch.tensor([51, 52]),
            "T": torch.tensor([0, 1, 2, 3, 4]),
            "S": torch.tensor([[0, 1], [0, 2], [0, 3]]),
            "L": torch.tensor([1, 1, 0, 1, 0]),
            "labels": torch.tensor([0, 0, 1, 1, 1]),
            "is_entity": torch.tensor([1, 1, 0, 0, 0]),
        }
        return sample_1

    @pytest.fixture
    def sample_2(self):
        sample_2 = {
            "tokens": torch.tensor([3, 4, 5, 6, 7, 8]),
            "entity_spans": torch.tensor([[0, -1], [1, 2]]),
            "element_names": torch.tensor([51, 52]),
            "T": torch.tensor([0, 1, 2, 3, 4]),
            "S": torch.tensor([[0, 1], [0, 2], [0, 3]]),
            "L": torch.tensor([1, 1, 0, 1, 0]),
            "labels": torch.tensor([0, 0, 1, 1, 1]),
            "is_entity": torch.tensor([1, 1, 0, 0, 0]),
        }
        return sample_2

    @pytest.fixture
    def collated(self, sample_1, sample_2):
        collated = collate_func([sample_1, sample_2])
        return collated

    list_keys = ["tokens", "entity_spans"]

    @pytest.mark.parametrize("key", list_keys)
    def test_collate_func_list_keys(self, key, sample_1, sample_2, collated):
        assert torch.all(collated[key][0] == sample_1[key]) and torch.all(
            collated[key][1] == sample_2[key]
        )

    cat_keys = ["element_names", "L", "labels", "is_entity", "L"]

    @pytest.mark.parametrize("key", cat_keys)
    def test_collate_func_cat_keys(self, key, sample_1, sample_2, collated):
        assert torch.all(collated[key] == torch.cat([sample_1[key], sample_2[key]]))

    def test_collate_func_T(self, sample_1, sample_2, collated):
        assert torch.all(
            collated["T"]
            == torch.arange(
                len(sample_1["element_names"]) + len(sample_2["element_names"])
            )
        )
