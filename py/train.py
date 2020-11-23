#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'nb_black')
import re
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
    CELL_STATE_CLAMP_VAL,
    HIDDEN_STATE_CLAMP_VAL,
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
    cell_state_clamp_val=CELL_STATE_CLAMP_VAL,
    hidden_state_clamp_val=HIDDEN_STATE_CLAMP_VAL,
)
param_names = [p[0] for p in model.named_parameters()]


# check not nan initialization

# In[4]:


for param in param_names:
    param = re.sub(r"\.([0-9])", r"[\1]", param)
    if torch.any(torch.isnan(eval(f"model.{param}"))):
        raise ValueError(f"param {param} initialized with nans")


# In[5]:


tb = SummaryWriter()

optimizer = torch.optim.Adadelta(
    model.parameters(), lr=1.0
)  # TODO: changed to see if solves nan attn_scores weights
criterion = nn.NLLLoss()

EPOCHS = 1

torch.autograd.set_detect_anomaly(True)

for epoch in range(EPOCHS):
    print([f"EPOCH {epoch}"])
    for step in tqdm(train_idx[0:3]):  # TODO: remove, added for testing
        n_iter = (epoch) * len(train_idx) + step
        sample = process_sample(dataset[step], dataset.inverse_schema)
        optimizer.zero_grad()
        raw_predictions = model(
            sample["tokens"],
            sample["entity_spans"],
            sample["element_names"],
            sample["H"],
            sample["A"],
            sample["T"],
            sample["S"],
        )
        predictions = torch.log(raw_predictions)
        loss = criterion(predictions, sample["labels"])
        if loss.isnan().item():
            print(raw_predictions)
            raise ValueError("NaN loss encountered")

        # if the model has made predictions on relations.
        # This doesn't happen if there are no possible relations in the sentence given the schema
        if len(predictions) > len(sample["entity_spans"]):
            loss.backward()
            optimizer.step()
            tb.add_scalar("loss", loss, n_iter)

        for param in param_names:
            param = re.sub(r"\.([0-9])", r"[\1]", param)
            tb.add_histogram(param, eval(f"model.{param}"), n_iter)
            tb.flush()

    with torch.no_grad():
        val_accs = []
        print([f"EPOCH {epoch} VALIDATION"])
        for step in tqdm(val_idx[0:3]):  # TODO: remove, added for testing
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

        val_acc = np.mean(val_accs)
        tb.add_scalar("val_acc", val_acc, n_iter)
        print("Epoch {:05d} | Val Acc {:.4f} |".format(epoch, val_acc))
        tb.flush()


# In[ ]:




