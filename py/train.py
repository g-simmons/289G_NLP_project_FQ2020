#!/usr/bin/env python
# coding: utf-8

import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

import wandb

sys.path.append("../py")
sys.path.append("../lib/BioInfer_software_1.0.1_Python3/")

from bioinferdataset import BioInferDataset
from config import *
from INN import INNModelLightning, collate_func


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
        GPUS = 1
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
        GPUS = 0
    return GPUS


def load_dataset():
    dataset = BioInferDataset(XML_PATH)
    if os.path.isfile(PREPPED_DATA_PATH):
        dataset.load_samples_from_pickle(PREPPED_DATA_PATH)
    else:
        dataset.prep_data()
        dataset.samples_to_pickle(PREPPED_DATA_PATH)
    return dataset


def split_data(dataset):
    train_max_range = round(0.7 * len(dataset))
    val_max_range = round(0.9 * len(dataset))
    test_max_range = len(dataset)

    train_idx = range(0, train_max_range)
    val_idx = range(train_max_range, val_max_range)
    test_idx= range(val_max_range,test_max_range)

    train_set, val_set, test_set = random_split(dataset, lengths=[len(train_idx), len(val_idx), len(test_idx)])

    return train_set, val_set, test_set


if __name__ == "__main__":
    GPUS = set_device()
    dataset = load_dataset()
    train_set, val_set, test_set = split_data(dataset)

    torch.autograd.set_detect_anomaly(True)

    train_data_loader = DataLoader(
        train_set,
        collate_fn=collate_func,
        batch_size=BATCH_SIZE,
        drop_last=True,
        shuffle=False,
    )
    val_data_loader = DataLoader(
        val_set, collate_fn=collate_func, batch_size=VAL_BATCH_SIZE, shuffle=VAL_SHUFFLE
    )
    test_data_loader = DataLoader(
        test_set, collate_fn=collate_func, batch_size=TEST_BATCH_SIZE,
    )

    run_name = "test"

    model = INNModelLightning(
        vocab_dict=dataset.vocab_dict,
        element_to_idx=dataset.element_to_idx,
        hidden_dim_bert=HIDDEN_DIM_BERT,
        output_bert_hidden_states=False,
        word_embedding_dim=WORD_EMBEDDING_DIM,
        cell_state_clamp_val=CELL_STATE_CLAMP_VAL,
        hidden_state_clamp_val=HIDDEN_STATE_CLAMP_VAL,
        encoding_method=ENCODING_METHOD,
    )

    wandb_config = {
        "GPUS": int(GPUS),
        "batch_size": BATCH_SIZE,
        "val_batch_size":VAL_BATCH_SIZE,
        "val_shuffle":VAL_SHUFFLE,
        "max_layers": MAX_LAYERS,
        "learning_rate": LEARNING_RATE,
        "cell_state_clamp_val": CELL_STATE_CLAMP_VAL,
        "hidden_state_clamp_val": HIDDEN_STATE_CLAMP_VAL,
        "word_embedding_dim": WORD_EMBEDDING_DIM,
        "exclude_samples": EXCLUDE_SAMPLES,
    }

    wandb_logger = WandbLogger(
        name=run_name,
        project="nested-relation-extraction",
        entity="ner",
        config=wandb_config,
        log_model=True,
    )
    wandb_logger.watch(model, log="gradients", log_freq=1)

    checkpoint_callback = ModelCheckpoint(dirpath=wandb.run.dir, save_top_k=-1)

    trainer = pl.Trainer(
        gpus=GPUS,
        progress_bar_refresh_rate=1,
        automatic_optimization=False,
        num_sanity_val_steps=2,
        max_epochs=3,
        val_check_interval=0.25,
        logger=wandb_logger,
        checkpoint_callback=checkpoint_callback, # save the model after each epoch
    )

    trainer.fit(model, train_data_loader, val_data_loader)

    test_results = trainer.test(test_dataloaders=test_data_loader)
    print(test_results)
