import os
import numpy as np
import warnings
import random
import torch
import torchaudio
import torch.nn as nn
from torch import optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import time
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateLogger,
)
from project.utils.functions import data_processing, GreedyDecoder, cer, wer
from pytorch_lightning.loggers import TensorBoardLogger

from project.model.las_test import Listener


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MAX_LENGTH = 2000

train_data = data.ConcatDataset(
    [
        torchaudio.datasets.LIBRISPEECH("data/", url=path, download=True)
        for path in ["dev-clean"]
    ]
)

data_loader = DataLoader(
    dataset=train_data,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda x: data_processing(x, "train"),
    num_workers=2,
)

print("Training...", flush=True)
teacher_forcing_ratio = 0.5

listener = Listener(128, 512, 3)

for batch_idx, _data in enumerate(data_loader):
    spectrograms, labels, input_lengths, label_lengths = _data
    print(labels[0])
    # listener(spectrograms)
