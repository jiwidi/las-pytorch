from pydub import AudioSegment
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import editdistance as ed
import pdb
from pydub import AudioSegment
import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import scipy.io.wavfile as wav
from python_speech_features import logfbank
import argparse
import re

##GLOBAL VARIABLES
IGNORE_ID = -1


def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))


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
            idx, char = line.strip().split(",")
            unit2idx[str(idx)] = char
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
    onehot_x = Variable(torch.LongTensor(batch_size, time_steps, encoding_dim).zero_().scatter_(-1, input_x, 1)).type(input_type)

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


def traverse(root, path, search_fix=".flac", return_label=False):
    f_list = []

    for p in path:
        p = root + p
        for sub_p in sorted(os.listdir(p)):
            for sub2_p in sorted(os.listdir(p + sub_p + "/")):
                if return_label:
                    # Read trans txt
                    with open(p + sub_p + "/" + sub2_p + "/" + sub_p + "-" + sub2_p + ".trans.txt", "r") as txt_file:
                        for line in txt_file:
                            f_list.append(" ".join(line[:-1].split(" ")[1:]))
                else:
                    # Read acoustic feature
                    for file in sorted(os.listdir(p + sub_p + "/" + sub2_p)):
                        if search_fix in file:
                            file_path = p + sub_p + "/" + sub2_p + "/" + file
                            f_list.append(file_path)
    return f_list


def flac2wav(f_path):
    flac_audio = AudioSegment.from_file(f_path, "flac")
    flac_audio.export(f_path[:-5] + ".wav", format="wav")


def mp32wav(f_path):
    mp3_audio = AudioSegment.from_mp3(f_path)
    mp3_audio.export(f_path[:-4] + ".wav", format="wav")


def wav2logfbank(f_path, win_size, n_filters, nfft=512):
    (rate, sig) = wav.read(f_path)
    fbank_feat = logfbank(sig, rate, winlen=win_size, nfilt=n_filters, nfft=nfft)
    os.remove(f_path)
    np.save(f_path[:-3] + "fb" + str(n_filters), fbank_feat)


def norm(f_path, mean, std):
    np.save(f_path, (np.load(f_path) - mean) / std)


def char_mapping(tr_text, target_path):
    char_map = {}
    char_map["<sos>"] = 0
    char_map["<eos>"] = 1
    char_idx = 2

    # map char to index
    for text in tr_text:
        for char in text:
            if char not in char_map:
                char_map[char] = char_idx
                char_idx += 1

    # Reverse mapping
    rev_char_map = {v: k for k, v in char_map.items()}

    # Save mapping
    with open(target_path + "idx2chap.csv", "w") as f:
        f.write("idx,char\n")
        for i in range(len(rev_char_map)):
            f.write(str(i) + "," + rev_char_map[i] + "\n")
    return char_map
