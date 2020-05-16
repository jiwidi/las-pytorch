import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from model.las_model import Listener, Speller
from torch.autograd import Variable
from data import AudioDataLoader, AudioDataset
from torch.utils.tensorboard import SummaryWriter
from solver.solver import batch_iterator
import numpy as np
import yaml
import os
import random
import enlighten
import argparse
import pdb

# Set cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Tensorboard logging
# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

parser = argparse.ArgumentParser(description="Training script for LAS on Librispeech .")
parser.add_argument(
    "--config_path", metavar="config_path", type=str, help="Path to config file for training.",
)
args = parser.parse_args()

# Fix seed
seed = 17
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print("---------------------------------------")
print("Loading Config...", flush=True)
# Load config file for experiment
config_path = args.config_path
print("Loading configure file at", config_path)
with open(config_path, "r") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

tf_rate_upperbound = params["training"]["tf_rate_upperbound"]
tf_rate_lowerbound = params["training"]["tf_rate_lowerbound"]
tf_decay_step = params["training"]["tf_decay_step"]
epochs = params["training"]["epochs"]

# Load datasets
print("---------------------------------------")
print("Processing datasets...", flush=True)
train_dataset = AudioDataset(params, "train")
train_loader = AudioDataLoader(train_dataset, num_workers=params["data"]["num_works"]).loader
dev_dataset = AudioDataset(params, "dev")
dev_loader = AudioDataLoader(dev_dataset, num_workers=params["data"]["num_works"]).loader

print("---------------------------------------")
print("Creating model architecture...", flush=True)
# Create listener and speller
listener = Listener(**params["model"]["listener"])
speller = Speller(**params["model"]["speller"])

# Create optimizer
optimizer = torch.optim.Adam(
    [{"params": listener.parameters()}, {"params": speller.parameters()}],
    lr=params["training"]["lr"],
)

print("---------------------------------------")
print("Training...", flush=True)

global_step = 0
for epoch in range(epochs):
    epoch_step = 0
    train_loss = []
    train_ler = []
    for i, (data) in enumerate(train_loader):
        print(
            f"Current Epoch: {epoch} | Epoch step: {epoch_step}/{len(train_loader)}",
            end="\r",
            flush=True,
        )
        # Adjust LR
        tf_rate = tf_rate_upperbound - (tf_rate_upperbound - tf_rate_lowerbound) * min(
            (float(global_step) / tf_decay_step), 1
        )

        inputs = data[1]["inputs"].to(device)
        labels = data[2]["targets"].to(device)

        # minibatch execution
        batch_loss, batch_ler = batch_iterator(
            batch_data=inputs,
            batch_label=labels,
            listener=listener,
            speller=speller,
            optimizer=optimizer,
            tf_rate=tf_rate,
            is_training=True,
            max_label_len=params["data"]["vocab_size"],
            label_smoothing=params["training"]["label_smoothing"],
        )
        train_loss.append(batch_loss)
        train_ler.extend(batch_ler)

        global_step += 1
        epoch_step += 1
        # print(batch_ler)

    train_loss = np.array([sum(train_loss) / len(train_loss)])
    train_ler = np.array([sum(train_ler) / len(train_ler)])
    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_scalar("cer/train", train_ler, epoch)
    # Validation
    val_loss = []
    val_ler = []
    for i, (data) in enumerate(train_loader):
        inputs = data[1]["inputs"].to(device)
        labels = data[2]["targets"].to(device)

        batch_loss, batch_ler = batch_iterator(
            batch_data=inputs,
            batch_label=labels,
            listener=listener,
            speller=speller,
            optimizer=optimizer,
            tf_rate=tf_rate,
            is_training=True,
            max_label_len=params["data"]["vocab_size"],
            label_smoothing=params["training"]["label_smoothing"],
        )
        val_loss.append(batch_loss)
        val_ler.extend(batch_ler)

    val_loss = np.array([sum(val_loss) / len(val_loss)])
    val_ler = np.array([sum(val_ler) / len(val_ler)])
    writer.add_scalar("loss/dev", val_loss, epoch)
    writer.add_scalar("cer/dev", val_ler, epoch)
    # writer.add_scalars("cer", {"train": np.array([np.array(batch_ler).mean()])}, global_step)
    # pdb.set_trace()
    # print(inputs.size())
