#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from utils import process_sample, get_child_indices


# In[2]:


dataset = BioInferDataset("../data/BioInfer_corpus_1.1.1.xml")

train_idx = range(0, 880)
val_idx = range(880, 990)


# In[3]:


model = INNModel(
    vocab_dict=dataset.vocab_dict,
    element_to_idx=dataset.element_to_idx,
    word_embedding_dim=WORD_EMBEDDING_DIM,
    relation_embedding_dim=RELATION_EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
)


# In[5]:


optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)
criterion = nn.NLLLoss()

EPOCHS = 1

for epoch in range(EPOCHS):
    for step in train_idx[0:2]:  # TODO: remove, added for testing
        sample = process_sample(dataset[step], dataset.inverse_schema)
        optimizer.zero_grad()
        predictions = torch.log(
            model(
                sample["tokens"],
                sample["entity_spans"],
                sample["element_names"],
                sample["H"],
                sample["A"],
                sample["T"],
                sample["S"],
            )
        )
        loss = criterion(predictions, sample["labels"])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_accs = []
        for step in val_idx[0:3]:  # TODO: remove, added for testing
            sample = process_sample(dataset[step], dataset.inverse_schema)
            labels = sample["labels"]
            predictions = torch.argmax(
                model(
                    sample["tokens"],
                    sample["entity_spans"],
                    sample["element_names"],
                    sample["H"],
                    sample["A"],
                    sample["T"],
                    sample["S"],
                )
            )
            acc = sum(predictions == labels) / len(labels)
            val_accs.append(acc.item())
        print("Epoch {:05d} | Val Acc {:.4f} |".format(epoch, np.mean(val_accs)))
