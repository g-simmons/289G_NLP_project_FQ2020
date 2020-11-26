import sys
import numpy as np
import torch
import dgl
from torch import nn
from torch.nn import functional as functional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append("../py")
sys.path.append("../lib/BioInfer_software_1.0.1_Python3/")

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

from bioinferdataset import BioInferDataset
from INN import INNModel


def get_child_indices(g, node_idx):
    return torch.stack(g.out_edges(node_idx))[1].tolist()


def process_sample(sample, inverse_schema):
    element_names = sample["element_names"].numpy()
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
                    element_names = np.append(element_names, predicate)
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

    sample["labels"] = torch.tensor(labels_temp, dtype=torch.long)
    sample["A"] = torch.tensor(A_temp)
    sample["T"] = torch.tensor(T_temp)
    sample["S"] = torch.stack(S_temp)
    sample["element_names"] = torch.tensor(element_names)
    sample["H"] = torch.randn(sample["A"].shape[0], 2 * HIDDEN_DIM)

    return sample
