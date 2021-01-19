import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from model.las_model import Listener, Speller, LAS
from utils.functions import purge
from torch.autograd import Variable
from utils.data import AudioDataLoader, AudioDataset
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
import sys
from tqdm import tqdm

# Set cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description="Training script for LAS on Librispeech .")
parser.add_argument(
    "--config_path", metavar="config_path", type=str, help="Path to config file for training.", required=True,
)
parser.add_argument(
    "--experiment_name", metavar="experiment_name", type=str, help="Name for tensorboard logs", default="",
)


def main(args):
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
    train_loader = AudioDataLoader(train_dataset, shuffle=True, num_workers=params["data"]["num_works"]).loader
    dev_dataset = AudioDataset(params, "test")
    dev_loader = AudioDataLoader(dev_dataset, num_workers=params["data"]["num_works"]).loader

    print("---------------------------------------")
    print("Creating model architecture...", flush=True)
    # Create listener and speller
    listener = Listener(**params["model"]["listener"])
    speller = Speller(**params["model"]["speller"])
    las = LAS(listener, speller)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        las = nn.DataParallel(las)
    print(las)
    las.cuda()
    # Create optimizer
    optimizer = torch.optim.Adam(params=las.parameters(), lr=params["training"]["lr"],)
    if params["training"]["continue_from"]:
        print("Loading checkpoint model %s" % params["training"]["continue_from"])
        package = torch.load(params["training"]["continue_from"])
        las.load_state_dict(package["state_dict"])
        optimizer.load_state_dict(package["optim_dict"])
        start_epoch = int(package.get("epoch", 1))
    else:
        start_epoch = 0

    print("---------------------------------------")
    print("Training...", flush=True)

    # import pdb

    # pdb.set_trace()
    global_step = 0 + (len(train_loader) * start_epoch)
    best_cv_loss = 10e5
    my_fields = {"loss": 0}
    for epoch in tqdm(range(start_epoch, epochs), desc="Epoch training"):
        epoch_step = 0
        train_loss = []
        train_ler = []
        batch_loss = 0
        for i, (data) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, desc=f"Epoch number {epoch}"):
            # print(
            #     f"Current Epoch: {epoch} Loss {np.round(batch_loss, 3)} | Epoch step: {epoch_step}/{len(train_loader)}",
            #     end="\r",
            #     flush=True,
            # )
            my_fields["loss"] = batch_loss
            # Adjust LR
            tf_rate = tf_rate_upperbound - (tf_rate_upperbound - tf_rate_lowerbound) * min(
                (float(global_step) / tf_decay_step), 1
            )
            inputs = data[1]["inputs"].cuda()
            labels = data[2]["targets"].cuda()
            print(f"INPUT SHAPE {inputs.shape} LABELS SHAPE: {labels.shape}")
            batch_loss, batch_ler = batch_iterator(
                batch_data=inputs,
                batch_label=labels,
                las_model=las,
                optimizer=optimizer,
                tf_rate=tf_rate,
                is_training=True,
                max_label_len=params["model"]["speller"]["max_label_len"],
                label_smoothing=params["training"]["label_smoothing"],
                vocab_dict=train_dataset.char2idx,
            )
            if i % 100 == 0:
                torch.cuda.empty_cache()
            train_loss.append(batch_loss)
            train_ler.extend(batch_ler)
            global_step += 1
            epoch_step += 1
            # print(batch_ler)
            writer.add_scalar("loss/train-step", batch_loss, global_step)
            writer.add_scalar("ler/train-step", np.array([sum(train_ler) / len(train_ler)]), global_step)

        train_loss = np.array([sum(train_loss) / len(train_loss)])
        train_ler = np.array([sum(train_ler) / len(train_ler)])
        writer.add_scalar("loss/train-epoch", train_loss, epoch)
        writer.add_scalar("ler/train-epoch", train_ler, epoch)
        # Validation
        val_loss = []
        val_ler = []
        val_step = 0
        for i, (data) in tqdm(enumerate(dev_loader), total=len(dev_loader), leave=False, desc="Validation"):
            # print(
            #     f"Current Epoch: {epoch} | Epoch step: {epoch_step}/{len(train_loader)} Validating step: {val_step}/{len(dev_loader)}",
            #     end="\r",
            #     flush=True,
            # )

            inputs = data[1]["inputs"].cuda()
            labels = data[2]["targets"].cuda()

            batch_loss, batch_ler = batch_iterator(
                batch_data=inputs,
                batch_label=labels,
                las_model=las,
                optimizer=optimizer,
                tf_rate=tf_rate,
                is_training=False,
                max_label_len=params["model"]["speller"]["vocab_size"],
                label_smoothing=params["training"]["label_smoothing"],
                vocab_dict=dev_dataset.char2idx,
            )
            if i % 100 == 0:
                torch.cuda.empty_cache()
            val_loss.append(batch_loss)
            val_ler.extend(batch_ler)
            val_step += 1

        val_loss = np.array([sum(val_loss) / len(val_loss)])
        val_ler = np.array([sum(val_ler) / len(val_ler)])
        writer.add_scalar("loss/dev", val_loss, epoch)
        writer.add_scalar("ler/dev", val_ler, epoch)
        # Checkpoint saving model each epoch and keeping only last 10 epochs
        if params["training"]["checkpoint"]:
            # Check if epoch-10 file exits, if so we delete it
            file_path_old = os.path.join(params["training"]["save_folder"], f"{data_name}-epoch{epoch - 10}.pth.tar")
            if os.path.exists(file_path_old):
                os.remove(file_path_old)

            file_path = os.path.join(params["training"]["save_folder"], f"{data_name}-epoch{epoch}.pth.tar")
            torch.save(
                las.serialize(optimizer=optimizer, epoch=epoch, tr_loss=val_loss, val_loss=val_loss), file_path,
            )
            print()
            print("Saving checkpoint model to %s" % file_path)

        if val_loss < best_cv_loss:  # We found a best model, lets save it too
            file_path = os.path.join(params["training"]["save_folder"], f"{data_name}-BEST_LOSS-epoch{epoch}.pth.tar")
            # purge(params["training"]["save_folder"], "*BEST_LOSS*")  # Remove
            # previous best models
            torch.save(
                las.serialize(optimizer=optimizer, epoch=epoch, tr_loss=val_loss, val_loss=val_loss), file_path,
            )
            print("Saving BEST model to %s" % file_path)

        # writer.add_scalars("cer", {"train": np.array([np.array(batch_ler).mean()])}, global_step)
        # pdb.set_trace()
        # print(inputs.size())
        print()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
