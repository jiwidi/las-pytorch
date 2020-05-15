import os
import torch
import numpy as np
import torchaudio as ta
from torch.utils.data import Dataset, DataLoader
import enlighten

PAD = 0
EOS = 1
BOS = 1
UNK = 2
MASK = 2
unk = '<UNK>'
compute_fbank = ta.compliance.kaldi.fbank


def load_vocab(vocab_file):
    unit2idx = {}
    with open(os.path.join(vocab_file), 'r', encoding='utf-8') as v:
        for line in v:
            unit, idx = line.strip().split()
            unit2idx[unit] = int(idx)
    return unit2idx


def normalization(feature):
    mean = torch.mean(feature)
    std = torch.std(feature)
    return (feature - mean) / std


def apply_cmvn(mat, stats):
    mean = stats[0, :-1] / stats[0, -1]
    variance = stats[1, :-1] / stats[0, -1] - np.square(mean)
    return np.divide(np.subtract(mat, mean), np.sqrt(variance))


def spec_augment(mel_spectrogram, frequency_mask_num=1, time_mask_num=2,
                 frequency_masking_para=5, time_masking_para=15):
    tau = mel_spectrogram.shape[0]
    v = mel_spectrogram.shape[1]

    warped_mel_spectrogram = mel_spectrogram

    # Step 2 : Frequency masking
    if frequency_mask_num > 0:
        for i in range(frequency_mask_num):
            f = np.random.uniform(low=0.0, high=frequency_masking_para)
            f = int(f)
            f0 = random.randint(0, v-f)
            warped_mel_spectrogram[:, f0:f0+f] = 0

    # Step 3 : Time masking
    if time_mask_num > 0:
        for i in range(time_mask_num):
            t = np.random.uniform(low=0.0, high=time_masking_para)
            t = int(t)
            t0 = random.randint(0, tau-t)
            warped_mel_spectrogram[t0:t0+t, :] = 0

    return warped_mel_spectrogram


def concat_and_subsample(features, left_frames=3, right_frames=0, skip_frames=2):

    time_steps, feature_dim = features.shape
    concated_features = np.zeros(
        shape=[time_steps, (1+left_frames+right_frames) * feature_dim], dtype=np.float32)

    concated_features[:, left_frames * feature_dim: (left_frames+1)*feature_dim] = features

    for i in range(left_frames):
        concated_features[i+1: time_steps, (left_frames-i-1)*feature_dim: (
            left_frames-i) * feature_dim] = features[0:time_steps-i-1, :]

    for i in range(right_frames):
        concated_features[0:time_steps-i-1, (right_frames+i+1)*feature_dim: (
            right_frames+i+2)*feature_dim] = features[i+1: time_steps, :]

    return concated_features[::skip_frames+1, :]


class AudioDataset(Dataset):
    def __init__(self, params, name='train'):

        self.params = params
        self.left_frames = params['left_frames']
        self.right_frames = params['right_frames']
        self.skip_frames = params['skip_frames']

        self.unit2idx = load_vocab(params['vocab'])

        self.targets_dict = {}
        self.file_list = []
        with open(params[name], 'r', encoding='utf-8') as t:
            next(t) #Skip headers line
            for line in t:
                parts = line.strip().split()
                sid = parts[0]
                path = parts[1]
                sentence = parts[2]
                label=[]
                for c in parts[1:]:
                    if c==' ':
                        c='<PAD>'
                    label.append(self.unit2idx[c] if c in self.unit2idx else self.unit2idx[unk])
                self.targets_dict[sid] = label
                self.file_list.append([sid, path])


        self.lengths = len(self.file_list)

    def verify_integrity_reads(self):
        pbar = enlighten.Counter(total=len(self.file_list), desc='Basic', unit='ticks')
        print("Verifying data integrity")
        for u in range(len(self.file_list)):
            utt_id, path = self.file_list[u]
            try:
                wavform, sample_frequency = ta.load(path)
            except:
                print(f"failed reading file at path {path} with index {u} and id {utt_id}")
            pbar.update()

    def __getitem__(self, index):
        utt_id, path = self.file_list[index]

        wavform, sample_frequency = ta.load(path)
        feature = compute_fbank(wavform, num_mel_bins=self.params['num_mel_bins'], sample_frequency=sample_frequency)

        if self.left_frames > 0 or self.right_frames > 0:
            feature = concat_and_subsample(feature, left_frames=self.left_frames,
                                           right_frames=self.right_frames, skip_frames=self.skip_frames)

        feature_length = feature.shape[0]
        targets = self.targets_dict[utt_id]
        targets_length = len(targets)

        return utt_id, feature, feature_length, targets, targets_length

    def __len__(self):
        return self.lengths

    def read_features(self, path):
        raise NotImplementedError

    def encode(self, seq):

        encoded_seq = []
        if self.encoding:
            for c in seq:
                if c in self.unit2idx:
                    encoded_seq.append(self.unit2idx[c])
                else:
                    encoded_seq.append(self.unit2idx['<UNK>'])
        else:
            encoded_seq = [int(i) for i in seq]

        return encoded_seq

    @property
    def idx2char(self):
        return {i: c for (c, i) in self.unit2idx.items()}

    @property
    def vocab_size(self):
        return len(self.unit2idx)

    @property
    def batch_size(self):
        return self.params['batch_size']


def collate_fn_with_eos_bos(batch):

    utt_ids = [data[0] for data in batch]
    features_length = [data[2] for data in batch]
    targets_length = [data[4] for data in batch]
    max_feature_length = max(features_length)
    max_target_length = max(targets_length)

    padded_features = []
    padded_targets = []

    for _, feat, feat_len, target, target_len in batch:
        padded_features.append(np.pad(feat, ((
            0, max_feature_length-feat_len), (0, 0)), mode='constant', constant_values=0.0))
        padded_targets.append(
            [BOS] + target + [EOS] + [PAD] * (max_target_length - target_len))

    features = torch.FloatTensor(padded_features)
    features_length = torch.IntTensor(features_length)
    targets = torch.LongTensor(padded_targets)
    targets_length = torch.IntTensor(targets_length)

    inputs = {
        'inputs': features,
        'inputs_length': features_length,
        'targets': targets,
        'targets_length': targets_length
    }
    return utt_ids, inputs


def collate_fn(batch):

    utt_ids = [data[0] for data in batch]
    features_length = [data[2] for data in batch]
    targets_length = [data[4] for data in batch]
    max_feature_length = max(features_length)
    max_target_length = max(targets_length)

    padded_features = []
    padded_targets = []

    for _, feat, feat_len, target, target_len in batch:
        padded_features.append(np.pad(feat, ((
            0, max_feature_length-feat_len), (0, 0)), mode='constant', constant_values=0.0))
        padded_targets.append(target + [PAD] * (max_target_length - target_len))

    features = torch.FloatTensor(padded_features)
    features_length = torch.IntTensor(features_length)
    targets = torch.LongTensor(padded_targets)
    targets_length = torch.IntTensor(targets_length)

    feature = {
        'inputs': features,
        'inputs_length': features_length
    }

    label = {
        'targets': targets,
        'targets_length': targets_length
    }
    return utt_ids, feature, label


class AudioDataLoader(object):
    def __init__(self, dataset, shuffle=False, ngpu=1, mode='ddp', include_eos_sos=False):
        if ngpu > 1:
            if mode == 'ddp':
                self.sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                self.sampler = None
        else:
            self.sampler = None

        self.loader = torch.utils.data.DataLoader(dataset, batch_size=dataset.batch_size * ngpu,
                                                  shuffle=shuffle if self.sampler is None else False,
                                                  num_workers=2 * ngpu, pin_memory=False, sampler=self.sampler,
                                                  collate_fn=collate_fn_with_eos_bos if include_eos_sos else collate_fn)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)