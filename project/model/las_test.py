from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 1400


class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout=0.0, batch_first=False):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=batch_first,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(rnn_dim)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = self.layer_norm(x)
        # x = F.gelu(x)
        x, _ = self.BiGRU(x)
        # x = self.dropout(x)
        return x


class Listener(nn.Module):
    def __init__(
        self, input_feature_dim_listener, hidden_size_listener, num_layers_listener
    ):
        super(Listener, self).__init__()
        assert num_layers_listener >= 1, "Listener should have at least 1 layer"
        self.hidden_size = hidden_size_listener
        self.embedding = nn.Embedding(
            input_feature_dim_listener,
            hidden_size_listener,
        )
        # self.gru = nn.Sequential(
        #     *[
        #         BidirectionalGRU(
        #             rnn_dim=input_feature_dim_listener
        #             if i == 1
        #             else int(hidden_size_listener / (2 * i)),
        #             hidden_size=int(hidden_size_listener / (2 * (i + 1))),
        #             dropout=0.1,
        #             batch_first=i == 1,
        #         )
        #         for i in range(1, num_layers_listener + 1)
        #     ]
        # )
        splitters = [1] + [i * 2 for i in range(1, num_layers_listener)]
        self.gru = nn.Sequential(
            BidirectionalGRU(
                rnn_dim=input_feature_dim_listener,
                hidden_size=hidden_size_listener,
                batch_first=True,
            ),
            *[
                BidirectionalGRU(
                    rnn_dim=int(hidden_size_listener / (i)) * 2,
                    hidden_size=int(int(hidden_size_listener / (i)) / 2),
                )
                for i in splitters
            ]
        )

    def forward(self, x):
        print(x.shape)
        x = x.squeeze().permute(0, 2, 1)
        print(x.shape)
        # embedded = self.embedding(input)  # .view(1, 1, -1)
        # output = embedded
        output = self.gru(x)
        return output  # , hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
