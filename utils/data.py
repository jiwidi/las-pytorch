import os
import torch
import numpy as np
import torchaudio as ta
from torch.utils.data import Dataset, DataLoader
import enlighten
import random
from .functions import load_vocab
import pdb

compute_fbank = ta.compliance.kaldi.fbank

PAD = 0
EOS = 1
BOS = 1
UNK = 2
MASK = 2
unk = "<UNK>"

listener_layers = 5


def OneHotEncode(idx, max_idx=42):
    idx = int(idx)
    new_y = np.zeros(max_idx)
    new_y[idx] = 1
    return new_y


def get_data(data_table, i):
    return np.load(data_table.loc[i]["input"])


class AudioDataset(Dataset):
    def __init__(self, params, name="train"):
        self.params = params
        # Vocabulary dictionary, character to idx
        self.char2idx = load_vocab(params["data"]["vocab"])
        self.batch_size = params["data"]["batch_size"]
        if name == "test":
            self.batch_size = 2
        listener_layers = params["model"]["listener"]["num_layers"]
        # The files paths and id
        self.targets_dict = {}
        self.file_list = []
        self.targets_real_target = {}
        with open(params["data"][name], "r", encoding="utf-8") as t:
            next(t)  # Skip headers line
            for line in t:
                parts = line.strip().split(",")
                sid = parts[0]
                path = parts[1]
                label = parts[2]

                self.targets_dict[sid] = label
                # self.targets_real_target[sid] = sentence
                self.file_list.append([sid, path])
                # print(f"For sentence: {len(sentence)}")
                # print(f"Generated   : {len(label)}")

        self.lengths = len(self.file_list)

    def __getitem__(self, index):
        utt_id, path = self.file_list[index]

        # REAL TIME FEATURE TRANSFORM IS THIS WAY LEL
        # wavform, sample_frequency = ta.load(path)
        # feature = compute_fbank(
        #     wavform,
        #     num_mel_bins=self.params["data"]["num_mel_bins"],
        #     sample_frequency=sample_frequency,
        # )
        feature = np.load(path)

        feature_length = feature.shape[0]
        targets = self.targets_dict[utt_id]
        targets = [OneHotEncode(index_char, self.params["data"]["vocab_size"]) for index_char in targets.strip().split(" ")]
        targets_length = [len(i) for i in targets]

        return utt_id, feature, feature_length, targets, targets_length

    def __len__(self):
        return self.lengths


def collate_fn_with_eos_bos(batch):
    utt_ids = [data[0] for data in batch]
    features_length = [data[2] for data in batch]
    targets_length = [data[4] for data in batch]
    max_feature_length = max(features_length)
    max_target_length = max(targets_length)

    padded_features = []
    padded_targets = []

    i = 0
    for _, feat, feat_len, target, target_len in batch:
        padded_features.append(np.pad(feat, ((0, max_feature_length - feat_len), (0, 0)), mode="constant", constant_values=0.0,))
        padded_targets.append([BOS] + target + [EOS] + [PAD] * (max_target_length - target_len[i]))
        i += 1

    features = torch.FloatTensor(padded_features)
    features_length = torch.IntTensor(features_length)
    targets = torch.LongTensor(padded_targets)
    targets_length = torch.IntTensor(targets_length)

    inputs = {
        "inputs": features,
        "inputs_length": features_length,
        "targets": targets,
        "targets_length": targets_length,
    }
    return utt_ids, inputs


def collate_fn(batch):

    utt_ids = [data[0] for data in batch]
    features_length = [data[2] for data in batch]
    targets_length = [data[4] for data in batch]
    targets_length = [len(i) for i in targets_length]
    max_feature_length = max(features_length)
    max_target_length = max(targets_length)
    if max_feature_length % (2 ** listener_layers) != 0:
        max_feature_length += (2 ** listener_layers) - (max_feature_length % (2 ** listener_layers))

    padded_features = []
    padded_targets = []

    i = 0
    for _, feat, feat_len, target, target_len in batch:
        padded_features.append(np.pad(feat, ((0, max_feature_length - feat_len), (0, 0)), mode="constant", constant_values=0.0,))
        tar_padds = [OneHotEncode(PAD, 30) for u in range((max_target_length - len(target)))]
        # print(len(target))
        # print(len(target[1]))
        padded_targets.append(target + tar_padds)
        # print(
        #     f"Maximum target is {max_target_length}, we have a target here of {len(target)} so we add a pad list of {len(tar_padds)} for a resultant {len(target + tar_padds)}"
        # )
        i += 1
    features = torch.FloatTensor(padded_features)
    features_length = torch.IntTensor(features_length)
    targets = torch.LongTensor(padded_targets)
    targets_length = torch.IntTensor(targets_length)

    feature = {"inputs": features, "inputs_length": features_length}

    label = {"targets": targets, "targets_length": targets_length}
    return utt_ids, feature, label


class AudioDataLoader(object):
    def __init__(
        self, dataset, shuffle=False, ngpu=1, mode="ddp", include_eos_sos=False, num_workers=8,
    ):
        if ngpu > 1:
            if mode == "ddp":
                self.sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                self.sampler = None
        else:
            self.sampler = None

        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset.batch_size * ngpu,
            shuffle=shuffle,
            num_workers=num_workers * ngpu,
            pin_memory=True,
            sampler=self.sampler,
            collate_fn=collate_fn_with_eos_bos if include_eos_sos else collate_fn,
        )

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
