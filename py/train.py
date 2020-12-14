#!/usr/bin/env python
# coding: utf-8

import sys
import os
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

sys.path.append("../py")
sys.path.append("../lib/BioInfer_software_1.0.1_Python3/")

from config import (
    WORD_EMBEDDING_DIM,
    RELATION_EMBEDDING_DIM,
    BATCH_SIZE,
    MAX_LAYERS,
    CELL_STATE_CLAMP_VAL,
    HIDDEN_STATE_CLAMP_VAL,
    XML_PATH,
    PREPPED_DATA_PATH,
    LEARNING_RATE,
    EXCLUDE_SAMPLES,
)

BATCH_SIZE = 2

from bioinferdataset import BioInferDataset
from INN import INNModelLightning

def collate_func(batch):
    cat_keys = ["element_names", "T", "L", "labels", "is_entity", "L"]
    list_keys = ["tokens", "entity_spans"]

    if type(batch) == dict:
        batch = [batch]
    new_batch = {}
    for key in cat_keys:
        new_batch[key] = torch.cat([sample[key] for sample in batch])
    for key in list_keys:
        new_batch[key] = [sample[key] for sample in batch]

    S = batch[0]["S"]
    for i in range(1, len(batch)):
        s_new = batch[i]["S"]
        s_new[s_new > -1] += batch[i - 1]["T"].shape[0]
        S = torch.cat([S, s_new])
    new_batch["S"] = S

    T = torch.arange(len(new_batch["S"]))
    new_batch["T"] = T

    return new_batch


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

    torch.autograd.set_detect_anomaly(True)

    train_set, val_set = random_split(dataset, lengths=[len(train_idx), len(val_idx)])

    train_data_loader = DataLoader(
        train_set, collate_fn=collate_func, batch_size=BATCH_SIZE, num_workers=4
    )
    val_data_loader = DataLoader(
        val_set, collate_fn=collate_func, batch_size=1, num_workers=4
    )

    model = INNModelLightning(
        vocab_dict=dataset.vocab_dict,
        element_to_idx=dataset.element_to_idx,
        word_embedding_dim=WORD_EMBEDDING_DIM,
        relation_embedding_dim=RELATION_EMBEDDING_DIM,
        cell_state_clamp_val=CELL_STATE_CLAMP_VAL,
        hidden_state_clamp_val=HIDDEN_STATE_CLAMP_VAL,
    )

    wandb_config = {
        "batch_size": BATCH_SIZE,
        "max_layers": MAX_LAYERS,
        "learning_rate": LEARNING_RATE,
        "cell_state_clamp_val": CELL_STATE_CLAMP_VAL,
        "hidden_state_clamp_val": HIDDEN_STATE_CLAMP_VAL,
        "word_embedding_dim": WORD_EMBEDDING_DIM,
        "relation_embedding_dim": RELATION_EMBEDDING_DIM,
        "exclude_samples": EXCLUDE_SAMPLES,
    }

    wandb_logger = WandbLogger(
        name="test",
        project="nested-relation-extraction",
        entity="ner",
        config=wandb_config,
        log_model=True,
    )
    wandb_logger.watch(model, log="gradients", log_freq=1)

    trainer = pl.Trainer(
        gpus=GPUS,
        progress_bar_refresh_rate=20,
        automatic_optimization=False,
        num_sanity_val_steps=2,
        max_epochs=3,
        val_check_interval=0.25,
        logger=wandb_logger,
    )

    trainer.fit(model, train_data_loader, val_data_loader)
