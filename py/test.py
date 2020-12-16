import pytest
import torch

from config import *
from daglstmcell import DAGLSTMCell
from train import collate_func
from INN import BERTEncoder


class TestINNModel:
    @pytest.fixture
    def seq_original(
        self,
    ):
        return [
            "alpha-catenin",
            "inhibits",
            "beta-catenin",
            "signaling",
            "by",
            "preventing",
            "formation",
            "of",
            "a",
            "beta-catenin*t-cell",
            "factor*dna",
            "complex",
            ".",
        ]

    # @pytest.fixture
    # def seq_bert(self,):
    #     return ['alpha', '-', 'cat', '##enin', 'inhibits', 'beta', '-', 'cat', '##enin', 'signaling', 'by', 'preventing', 'formation', 'of', 'a', 'beta', '-', 'cat', '##enin', '*', 't', '-', 'cell', 'factor', '*', 'dna', 'complex', '.']

    @pytest.fixture
    def offset_mapping(
        self,
    ):
        return [
            (0, 5),
            (5, 6),
            (6, 9),
            (9, 13),
            (0, 8),
            (0, 4),
            (4, 5),
            (5, 8),
            (8, 12),
            (0, 9),
            (0, 2),
            (0, 10),
            (0, 9),
            (0, 2),
            (0, 1),
            (0, 4),
            (4, 5),
            (5, 8),
            (8, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (15, 19),
            (0, 6),
            (6, 7),
            (7, 10),
            (0, 7),
            (0, 1),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ]

    @pytest.fixture
    def bert_tokens(
        self,
    ):
        return torch.tensor(
            [
                [
                    102,
                    6010,
                    579,
                    1793,
                    12280,
                    9233,
                    6130,
                    579,
                    1793,
                    12280,
                    3354,
                    214,
                    9778,
                    2256,
                    131,
                    106,
                    6130,
                    579,
                    1793,
                    12280,
                    1375,
                    105,
                    579,
                    377,
                    1491,
                    1375,
                    1732,
                    1127,
                    205,
                    103,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ]
        ).unsqueeze(0)

    @pytest.fixture
    def mask(self):
        return torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0]]).unsqueeze(0)

    @pytest.fixture
    def text(self):
        return ['alpha-catenin inhibits beta-catenin signaling by preventing formation of a beta-catenin*T-cell factor*DNA complex .']

    @pytest.fixture
    def stored_forward_results(self):
        return torch.load('../data/unit_tests/bert_out.tensor')

    def test_bert_parsing_integrated(self, seq_original, offset_mapping):
        bert_encodings = torch.randn(1, 28, 768)
        split = BERTEncoder.parse_bert(seq_original, offset_mapping)
        new_out = BERTEncoder.bert_new_embedding(bert_encodings, split)
        assert new_out.shape == (1, 13, 768)
        assert torch.all(
            (new_out[0, 0, :] - bert_encodings[0, 0:4, :].mean(dim=0)) == 0
        )

    def test_bert_forward(self, bert_tokens, mask, text, stored_forward_results):
        bert_enc = BERTEncoder(output_bert_hidden_states=False)
        bert_outs, _ = bert_enc.forward(bert_tokens, mask, text)
        assert (torch.all((bert_outs[0] - stored_forward_results) < 1e-5)) # not exact because results were saved to file




# class TestDAGLSTM:
#     def test_dag_lstm_forward(self):
#         MAX_INPUTS = 2
#         HIDDEN_DIM = 512

#         cell = DAGLSTMCell(
#             blstm_out_dim=HIDDEN_DIM,
#             max_inputs=MAX_INPUTS,
#             hidden_state_clamp_val=HIDDEN_STATE_CLAMP_VAL,
#             cell_state_clamp_val=CELL_STATE_CLAMP_VAL,
#         )

#         H = torch.randn(10 * MAX_INPUTS, HIDDEN_DIM).reshape(10, -1)
#         C = torch.zeros_like(H)
#         S = torch.arange(0, 10 * MAX_INPUTS).reshape(10, MAX_INPUTS)
#         E = torch.randn(10, HIDDEN_DIM)

#         h, c = cell.forward(H, C, E, S)
#         assert tuple(h.shape) == (10, HIDDEN_DIM)
#         assert tuple(c.shape) == (10, HIDDEN_DIM)


# class TestTrain:
#     @pytest.fixture
#     def sample_1(self):
#         sample_1 = {
#             "tokens": torch.tensor([3, 4, 5, 6, 7, 8]),
#             "entity_spans": torch.tensor([[0, -1], [1, 2]]),
#             "element_names": torch.tensor([51, 52]),
#             "T": torch.tensor([0, 1, 2, 3, 4]),
#             "S": torch.tensor([[0, 1], [0, 2], [0, 3]]),
#             "L": torch.tensor([1, 1, 0, 1, 0]),
#             "labels": torch.tensor([0, 0, 1, 1, 1]),
#             "is_entity": torch.tensor([1, 1, 0, 0, 0]),
#         }
#         return sample_1

#     @pytest.fixture
#     def sample_2(self):
#         sample_2 = {
#             "tokens": torch.tensor([3, 4, 5, 6, 7, 8]),
#             "entity_spans": torch.tensor([[0, -1], [1, 2]]),
#             "element_names": torch.tensor([51, 52]),
#             "T": torch.tensor([0, 1, 2, 3, 4]),
#             "S": torch.tensor([[0, 1], [0, 2], [0, 3]]),
#             "L": torch.tensor([1, 1, 0, 1, 0]),
#             "labels": torch.tensor([0, 0, 1, 1, 1]),
#             "is_entity": torch.tensor([1, 1, 0, 0, 0]),
#         }
#         return sample_2

#     @pytest.fixture
#     def collated(self, sample_1, sample_2):
#         collated = collate_func([sample_1, sample_2])
#         return collated

#     list_keys = ["tokens", "entity_spans"]

#     @pytest.mark.parametrize("key", list_keys)
#     def test_collate_func_list_keys(self, key, sample_1, sample_2, collated):
#         assert torch.all(collated[key][0] == sample_1[key]) and torch.all(
#             collated[key][1] == sample_2[key]
#         )

#     cat_keys = ["element_names", "L", "labels", "is_entity", "L"]

#     @pytest.mark.parametrize("key", cat_keys)
#     def test_collate_func_cat_keys(self, key, sample_1, sample_2, collated):
#         assert torch.all(collated[key] == torch.cat([sample_1[key], sample_2[key]]))

#     def test_collate_func_T(self, sample_1, sample_2, collated):
#         assert torch.all(
#             collated["T"]
#             == torch.arange(
#                 len(sample_1["element_names"]) + len(sample_2["element_names"])
#             )
#         )
