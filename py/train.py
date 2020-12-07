#!/usr/bin/env python
# coding: utf-8

import re
import sys
import os
import numpy as np
import torch
import dgl
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as functional
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from tqdm import tqdm
from pathlib import Path

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
    XML_PATH,
    PREPPED_DATA_PATH
)

from bioinferdataset import BioInferDataset
from INN import INNModelLightning
from daglstmcell import DAGLSTMCell

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
        GPUS = 1
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
        GPUS = 0

    dataset = BioInferDataset(XML_PATH)
    if os.path.isfile(PREPPED_DATA_PATH):
        dataset.load_samples_from_pickle(PREPPED_DATA_PATH)
    else:
        dataset.prep_data()
        dataset.samples_to_pickle(PREPPED_DATA_PATH)

    train_max_range = round(0.8 * len(dataset))
    train_idx = range(0, train_max_range)
    val_idx = range(train_max_range, len(dataset))

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
        t_list = []
        s_list = []
        labels_list = []

        # for each dataset entry selected to be in the batch
        for index in range(len(data)):
            # extract the entry's sample
            curr_sample = data[index]

            # put the sample's contents into its appropriate list
            token_list.append(curr_sample["tokens"])
            entity_spans_list.append(curr_sample["entity_spans"].long())
            element_names_list.append(torch.flatten(curr_sample["element_names"]))
            t_list.append(curr_sample["T"])
            s_list.append(curr_sample["S"])
            labels_list.append(curr_sample["labels"])

        # pads each sample content accordingly
        # format is L, B, D where L is the longest length, B is the batch size, D is the dimension size
        # sorts entries in token list in descending order
        token_len_list = torch.LongTensor([len(entry) for entry in token_list])
        token_len_list, argsort_list = token_len_list.sort(dim=0, descending=True)
        token_list = [token_list[index] for index in argsort_list]

        # sorts other inputs accordingly to how the token list was sorted
        entity_spans_list = [entity_spans_list[index] for index in argsort_list]
        element_names_list = [element_names_list[index] for index in argsort_list]
        t_list = [t_list[index] for index in argsort_list]
        s_list = [s_list[index] for index in argsort_list]
        labels_list = [labels_list[index] for index in argsort_list]

        the_batch_sample["tokens_pre-padded_size"] = token_len_list
        the_batch_sample["labels"] = torch.cat(labels_list, dim=0)
        the_batch_sample["entity_spans_pre-padded_size"] = [len(entry) for entry in entity_spans_list]

        # if the batch size is 1, then we need to add a dimension of size 1 to represent the batch size
        if len(data) == 1:
            the_batch_sample["tokens"] = token_list[0].unsqueeze(1)
            the_batch_sample["entity_spans"] = entity_spans_list[0].unsqueeze(1)
            the_batch_sample["element_names"] = element_names_list[0].unsqueeze(1)
            the_batch_sample["T"] = t_list[0].unsqueeze(1)
            the_batch_sample["S"] = s_list[0].unsqueeze(1)

        # if the batch is > 1, then we need to pad the input so that they all have the same dimensions
        else:
            the_batch_sample["tokens"] = pad_sequence(token_list, padding_value=dataset.vocab_dict["UNK"])
            the_batch_sample["entity_spans"] = pad_sequence(entity_spans_list, padding_value=-1)
            the_batch_sample["element_names"] = pad_sequence(element_names_list, padding_value=-1)
            the_batch_sample["T"] = pad_sequence(t_list, padding_value=0)
            the_batch_sample["S"] = pad_sequence(s_list, padding_value=0)

        return the_batch_sample

    # splits the dataset into a training set and test set
    train_set, val_set = random_split(dataset, lengths=[len(train_idx), len(val_idx)])

    # iterators that automatically give you the next batched samples using the collate function
    train_data_loader = DataLoader(train_set, collate_fn=collate_func, batch_size=BATCH_SIZE)
    val_data_loader = DataLoader(val_set, collate_fn=collate_func, batch_size=1)


    model = INNModelLightning(vocab_dict=dataset.vocab_dict,
            element_to_idx=dataset.element_to_idx,
            word_embedding_dim=WORD_EMBEDDING_DIM,
            relation_embedding_dim=RELATION_EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            cell_state_clamp_val=CELL_STATE_CLAMP_VAL,
            hidden_state_clamp_val=HIDDEN_STATE_CLAMP_VAL,)

    trainer = pl.Trainer(gpus=GPUS,
                        progress_bar_refresh_rate=1,
                        automatic_optimization=False,
                        max_steps=1,
                        profiler="advanced")
    trainer.fit(model, train_data_loader, val_data_loader)

    # # for each epoch
    # for epoch in range(EPOCHS):
    #     print([f"EPOCH {epoch}"])

    #     # iterate over the data loader; data loader gives next batched sample
    #     for step, batch_sample in enumerate(train_data_loader):
    #         n_iter = (epoch) * len(train_idx) + step
    #         optimizer.zero_grad()

    #         # does a forward pass with the model using the batched sample as input
    #         raw_predictions = model(
    #             batch_sample["tokens"],
    #             batch_sample["entity_spans"],
    #             batch_sample["element_names"],
    #             batch_sample["H"],
    #             batch_sample["T"],
    #             batch_sample["S"],
    #             batch_sample["entity_spans_pre-padded_size"],
    #             batch_sample["tokens_pre-padded_size"],
    #         )

    #         predictions = torch.log(raw_predictions)
    #         loss = criterion(predictions, batch_sample["labels"])
    #         if loss.isnan().item():
    #             print(raw_predictions)
    #             raise ValueError("NaN loss encountered")

    #         # keeps track of how many rows in the entity spans there are in total
    #         entity_spans_total_num = sum(batch_sample["entity_spans_pre-padded_size"])

    #         # if the model has made predictions on relations.
    #         # This doesn't happen if there are no possible relations in the sentence given the schema
    #         if len(predictions) > entity_spans_total_num:
    #             loss.backward()
    #             optimizer.step()
    #             tb.add_scalar("loss", loss, n_iter)

    #         for param in param_names:
    #             param = re.sub(r"\.([0-9])", r"[\1]", param)
    #             tb.add_histogram(param, eval(f"model.{param}"), n_iter)
    #             tb.flush()

    #     # validation phase; parameters will not be updated during this time
    #     # currently uses a batch size of 1
    #     with torch.no_grad():
    #         val_accs = []

    #         print([f"EPOCH {epoch} VALIDATION"])
    #         for step, batch_sample in enumerate(val_data_loader):
    #             labels = batch_sample["labels"]
    #             predictions = torch.argmax(
    #                 model(
    #                     batch_sample["tokens"],
    #                     batch_sample["entity_spans"],
    #                     batch_sample["element_names"],
    #                     batch_sample["H"],
    #                     batch_sample["T"],
    #                     batch_sample["S"],
    #                     batch_sample["entity_spans_pre-padded_size"],
    #                     batch_sample["tokens_pre-padded_size"],
    #                 )
    #             )
    #             acc = sum(predictions == labels) / len(labels)
    #             val_accs.append(acc.item())

    #         val_acc = np.mean(val_accs)
    #         tb.add_scalar("val_acc", val_acc, n_iter)
    #         print("Epoch {:05d} | Val Acc {:.4f} |".format(epoch, val_acc))
    #         tb.flush()




