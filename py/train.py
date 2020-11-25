#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import sys
import numpy as np
import torch
import dgl
from torch import nn
from torch.nn import functional as functional
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, random_split
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


def collate_func(data):
    """
    custom collate function; makes each sample's input have the same dimension as the longest one
    input consists of the dataset entries that were automatically selected to be in the next batch
    """

    the_batch_sample = dict()

    token_list = []
    entity_spans_list = []
    element_names_list = []
    h_list = []
    a_list = []
    t_list = []
    s_list = []
    labels_list = []

    # for each dataset entry selected to be in the batch
    for index in range(len(data)):
        # extract the entry's sample
        curr_sample = process_sample(data[index], dataset.inverse_schema)

        # put the sample's contents into its appropriate list
        token_list.append(curr_sample["tokens"])
        entity_spans_list.append(curr_sample["entity_spans"])
        element_names_list.append(torch.flatten(curr_sample["element_names"]))  # TODO: FIX IF FLATTENING IS WRONG
        h_list.append(curr_sample["H"])
        a_list.append(curr_sample["A"])
        t_list.append(curr_sample["T"])
        s_list.append(curr_sample["S"])
        labels_list.append(curr_sample["labels"])

    # pads each sample content accordingly
    # format is L, B, D where L is the longest length, B is the batch size, D is the dimension size

    # pads the tokens with UNK's index
    the_batch_sample["tokens"] = pad_sequence(token_list, padding_value=dataset.vocab_dict["UNK"])
    the_batch_sample["entity_spans"] = pad_sequence(entity_spans_list, padding_value=-1)
    the_batch_sample["element_names"] = pad_sequence(element_names_list, padding_value=-1)
    the_batch_sample["H"] = pad_sequence(h_list, padding_value=0)
    the_batch_sample["A"] = pad_sequence(a_list, padding_value=0)
    the_batch_sample["T"] = pad_sequence(t_list, padding_value=0)
    the_batch_sample["S"] = pad_sequence(s_list, padding_value=0)
    the_batch_sample["labels"] = pad_sequence(labels_list, padding_value=0)
    return the_batch_sample


# splits the dataset into a training set and test set
# TODO: val_idx + train_idx does not add up to the entire dataset (1,100)
train_set, test_set = random_split(dataset, lengths=[len(train_idx), len(dataset) - len(train_idx)])

# iterator that automatically gives you the next batched samples using the collate function
data_loader = DataLoader(train_set, collate_fn=collate_func, batch_size=BATCH_SIZE)

# training loop

# for each epoch
for epoch in range(EPOCHS):
    print([f"EPOCH {epoch}"])

    # iterate over the data loader; data loader gives next batched sample
    for step, batch_sample in enumerate(data_loader):
        n_iter = (epoch) * len(train_idx) + step
        optimizer.zero_grad()

        # does a forward pass with the model using the batched sample as input
        raw_predictions = model(
            batch_sample["tokens"],
            batch_sample["entity_spans"],
            batch_sample["element_names"],
            batch_sample["H"],
            batch_sample["A"],
            batch_sample["T"],
            batch_sample["S"],
        )

        predictions = torch.log(raw_predictions)
        loss = criterion(predictions, batch_sample["labels"])
        if loss.isnan().item():
            print(raw_predictions)
            raise ValueError("NaN loss encountered")

        # if the model has made predictions on relations.
        # This doesn't happen if there are no possible relations in the sentence given the schema
        if len(predictions) > len(batch_sample["entity_spans"]):  # TODO: NO LONGER WORKS CORRECTLY
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




