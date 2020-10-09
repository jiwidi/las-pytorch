"""
Example template for defining a system.
"""
from argparse import ArgumentParser

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchaudio
import torchvision.transforms as transforms
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader

from project.utils.functions import data_processing, GreedyDecoder, cer, wer
from project.utils.cosine_annearing_with_warmup import CosineAnnealingWarmUpRestarts


class Listener(nn.Module):
    def __init__(
        self, input_feature_dim_listener, hidden_size_listener, num_layers_listener
    ):
        super(Listener, self).__init__()
        self.hidden_size = hidden_size_listener

        self.embedding = nn.Embedding(input_feature_dim_listener, hidden_size_listener)
        # self.gru = nn.GRU(hidden_size, hidden_size)
        self.gru = nn.Sequential(
            nn.GRU(hidden_size_listener, hidden_size_listener / 2),
            *[
                nn.GRU(
                    hidden_size_listener / (2 * i), hidden_size_listener / (2 * (i + 1))
                )
                for _, i in enumerate(range(num_layers_listener - 1), 1)
            ]
        )

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class Speller(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(Speller, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1
        )
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)
        )

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class LAS(LightningModule):
    def __init__(
        self,
        input_feature_dim_listener,
        hidden_size_listener,
        num_layers_listener,
        input_feature_dim_speller,
        hidden_size_speller,
        num_layers_speller,
        dropout=0.1,
        **kwargs
    ):
        super(LAS, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = self.hparams.learning_rate

        self.listener = Listener(
            input_feature_dim_listener, hidden_size_listener, num_layers_listener
        )
        self.speller = Speller(
            input_feature_dim_speller,
            hidden_size_speller,
            num_layers_speller,
        )

        self.criterion = nn.CTCLoss(blank=28)
        self.example_input_array = torch.rand(8, 1, 128, 1151)

    def forward(self, x):
        # visit https://colab.research.google.com/drive/1JwtoPGdLFI9aM2kXAkj-P8h-GvnXJieY#scrollTo=AoO-CiW6nl23
        return x

    def serialize(self, optimizer, epoch, tr_loss, val_loss):
        package = {
            "state_dict": self.state_dict(),
            "optim_dict": optimizer.state_dict(),
            "epoch": epoch,
        }
        if tr_loss is not None:
            package["tr_loss"] = tr_loss
            package["val_loss"] = val_loss
        return package

    # ---------------------
    # Pytorch lightning overrides
    # ---------------------
    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        spectrograms, labels, input_lengths, label_lengths = batch
        y_hat = self(spectrograms)
        output = F.log_softmax(y_hat, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        tensorboard_logs = {"Loss/train": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        spectrograms, labels, input_lengths, label_lengths = batch
        y_hat = self(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(y_hat, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = self.criterion(output, labels, input_lengths, label_lengths)

        decoded_preds, decoded_targets = GreedyDecoder(
            output.transpose(0, 1), labels, label_lengths
        )
        n_correct_pred = sum(
            [int(a == b) for a, b in zip(decoded_preds, decoded_targets)]
        )

        test_cer, test_wer = [], []
        for j in range(len(decoded_preds)):
            test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
            test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

        avg_cer = torch.FloatTensor([sum(test_cer) / len(test_cer)])
        avg_wer = torch.FloatTensor(
            [sum(test_wer) / len(test_wer)]
        )  # Need workt to make all operations in torch
        logs = {
            "cer": avg_cer,
            "wer": avg_wer,
        }
        return {
            "val_loss": loss,
            "n_correct_pred": n_correct_pred,
            "n_pred": len(spectrograms),
            "log": logs,
            "wer": avg_wer,
            "cer": avg_cer,
        }

    def test_step(self, batch, batch_idx):
        spectrograms, labels, input_lengths, label_lengths = batch
        y_hat = self(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(y_hat, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = self.criterion(output, labels, input_lengths, label_lengths)

        decoded_preds, decoded_targets = GreedyDecoder(
            output.transpose(0, 1), labels, label_lengths
        )
        n_correct_pred = sum(
            [int(a == b) for a, b in zip(decoded_preds, decoded_targets)]
        )

        test_cer, test_wer = [], []
        for j in range(len(decoded_preds)):
            test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
            test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

        avg_cer = torch.FloatTensor([sum(test_cer) / len(test_cer)])
        avg_wer = torch.FloatTensor(
            [sum(test_wer) / len(test_wer)]
        )  # Need workt to make all operations in torch
        logs = {
            "Metrics/cer": avg_cer,
            "Metrics/wer": avg_wer,
        }
        return {
            "val_loss": loss,
            "n_correct_pred": n_correct_pred,
            "n_pred": len(spectrograms),
            "log": logs,
            "wer": avg_wer,
            "cer": avg_cer,
        }

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(
            x["n_pred"] for x in outputs
        )
        avg_wer = torch.stack([x["wer"] for x in outputs]).mean()
        avg_cer = torch.stack([x["cer"] for x in outputs]).mean()
        tensorboard_logs = {
            "Loss/val": avg_loss,
            "val_acc": val_acc,
            "Metrics/wer": avg_wer,
            "Metrics/cer": avg_cer,
        }
        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
            "wer": avg_wer,
            "cer": avg_cer,
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(
            x["n_pred"] for x in outputs
        )
        avg_wer = torch.stack([x["wer"] for x in outputs]).mean()
        avg_cer = torch.stack([x["cer"] for x in outputs]).mean()
        tensorboard_logs = {
            "Loss/test": avg_loss,
            "test_acc": test_acc,
            "Metrics/wer": avg_wer,
            "Metrics/cer": avg_cer,
        }
        return {
            "test_loss": avg_loss,
            "log": tensorboard_logs,
            "wer": avg_wer,
            "cer": avg_cer,
        }

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate / 10)
        # lr_scheduler = {'scheduler':optim.lr_scheduler.CyclicLR(optimizer,base_lr=self.hparams.learning_rate/5,max_lr=self.hparams.learning_rate,step_size_up=2000,cycle_momentum=False),
        lr_scheduler = {  # 'scheduler': optim.lr_scheduler.OneCycleLR(
            #     optimizer,
            #     max_lr=self.learning_rate,
            #     steps_per_epoch=int(len(self.train_dataloader())),
            #     epochs=self.hparams.epochs,
            #     anneal_strategy="linear",
            #     final_div_factor = 0.06,
            #     pct_start = 0.04
            # ),
            "scheduler": CosineAnnealingWarmUpRestarts(
                optimizer,
                T_0=int(len(self.train_dataloader()) * math.pi),
                T_mult=2,
                eta_max=self.learning_rate,
                T_up=int(len(self.train_dataloader())) * 2,
                gamma=0.8,
            ),
            "name": "learning_rate",  # Name for tensorboard logs
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler]

    def prepare_data(self):
        a = [
            torchaudio.datasets.LIBRISPEECH(
                self.hparams.data_root, url=path, download=True
            )
            for path in self.hparams.data_train
        ]
        b = [
            torchaudio.datasets.LIBRISPEECH(
                self.hparams.data_root, url=path, download=True
            )
            for path in self.hparams.data_test
        ]
        return a, b

    def setup(self, stage):
        self.train_data = data.ConcatDataset(
            [
                torchaudio.datasets.LIBRISPEECH(
                    self.hparams.data_root, url=path, download=True
                )
                for path in self.hparams.data_train
            ]
        )
        self.test_data = data.ConcatDataset(
            [
                torchaudio.datasets.LIBRISPEECH(
                    self.hparams.data_root, url=path, download=True
                )
                for path in self.hparams.data_test
            ]
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=lambda x: data_processing(x, "train"),
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=lambda x: data_processing(x, "valid"),
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=lambda x: data_processing(x, "valid"),
            num_workers=self.hparams.num_workers,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument("--n_cnn_layers", default=3, type=int)
        parser.add_argument("--n_rnn_layers", default=5, type=int)
        parser.add_argument("--rnn_dim", default=512, type=int)
        parser.add_argument("--n_class", default=29, type=int)
        parser.add_argument("--n_feats", default=128, type=str)
        parser.add_argument("--stride", default=2, type=int)
        parser.add_argument("--dropout", default=0.1, type=float)

        return parser
