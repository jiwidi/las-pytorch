import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from model.las_model import Listener, Speller, LAS
from utils.utils import purge
from torch.autograd import Variable
from data import AudioDataLoader, AudioDataset
from torch.utils.tensorboard import SummaryWriter
from solver.solver import batch_iterator
from sys import getsizeof
import numpy as np
import yaml
import os
import random
import enlighten
import argparse
import pdb

# Set cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description="Training script for LAS on Librispeech .")
parser.add_argument(
    "--config_path",
    metavar="config_path",
    type=str,
    help="Path to config file for training.",
    required=True,
)
parser.add_argument(
    "--experiment_name",
    metavar="experiment_name",
    type=str,
    help="Name for tensorboard logs",
    default="",
)
args = parser.parse_args()

# Tensorboard logging
# Writer will output to ./runs/ directory by default
writer = SummaryWriter(comment=args.experiment_name)
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
data_name = params["data"]["name"]

tf_rate_upperbound = params["training"]["tf_rate_upperbound"]
tf_rate_lowerbound = params["training"]["tf_rate_lowerbound"]
tf_decay_step = params["training"]["tf_decay_step"]
epochs = params["training"]["epochs"]

# Load datasets
print("---------------------------------------")
print("Processing datasets...", flush=True)
train_dataset = AudioDataset(params, "train")
train_loader = AudioDataLoader(
    train_dataset, shuffle=True, num_workers=params["data"]["num_works"]
).loader
dev_dataset = AudioDataset(params, "dev")
dev_loader = AudioDataLoader(dev_dataset, num_workers=params["data"]["num_works"]).loader


for i, (data) in enumerate(train_loader):
    pass
