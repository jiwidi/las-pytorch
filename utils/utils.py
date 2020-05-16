from pydub import AudioSegment
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import editdistance as ed
import pdb

##GLOBAL VARIABLES
IGNORE_ID = -1


def pad_list(xs, pad_value):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]
    return pad


def load_vocab(vocab_file):
    unit2idx = {}
    with open(os.path.join(vocab_file), "r", encoding="utf-8") as v:
        for line in v:
            unit, idx = line.strip().split()
            unit2idx[unit] = int(idx)
    return unit2idx


# CreateOnehotVariable function
# *** DEV NOTE : This is a workaround to achieve one, I'm not sure how this function affects the training speed ***
# This is a function to generate an one-hot encoded tensor with given batch size and index
# Input : input_x which is a Tensor or Variable with shape [batch size, timesteps]
#         encoding_dim, the number of classes of input
# Output: onehot_x, a Variable containing onehot vector with shape [batch size, timesteps, encoding_dim]
def CreateOnehotVariable(input_x, encoding_dim=63):
    if type(input_x) is Variable:
        input_x = input_x.data
    input_type = type(input_x)
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    input_x = input_x.unsqueeze(2).type(torch.LongTensor)
    onehot_x = Variable(
        torch.LongTensor(batch_size, time_steps, encoding_dim)
        .zero_()
        .scatter_(-1, input_x, 1)
    ).type(input_type)

    return onehot_x


# TimeDistributed function
# This is a pytorch version of TimeDistributed layer in Keras I wrote
# The goal is to apply same module on each timestep of every instance
# Input : module to be applied timestep-wise (e.g. nn.Linear)
#         3D input (sequencial) with shape [batch size, timestep, feature]
# output: Processed output      with shape [batch size, timestep, output feature dim of input module]
def TimeDistributed(input_module, input_x):
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    reshaped_x = input_x.contiguous().view(-1, input_x.size(-1))
    output_x = input_module(reshaped_x)
    return output_x.view(batch_size, time_steps, -1)
