# This preprocess scripts was made by https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch, I just edited some stuff
from pydub import AudioSegment
import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import scipy.io.wavfile as wav
from python_speech_features import logfbank
from functions import traverse, wav2logfbank, mp32wav, norm, char_mapping
import argparse
import pandas as pd
import unicodedata

parser = argparse.ArgumentParser(description="Common-voice preprocess.")

parser.add_argument(
    "--root",
    metavar="root",
    type=str,
    required=True,
    help="Absolute file path to Common voice dataset. (e.g. /usr/downloads/mozilla/)",
)

parser.add_argument(
    "--n_jobs",
    dest="n_jobs",
    action="store",
    default=-2,
    help="number of cpu availible for preprocessing.\n -1: use all cpu, -2: use all cpu but one",
)
parser.add_argument(
    "--n_filters",
    dest="n_filters",
    action="store",
    default=40,
    help="number of filters for fbank. (Default : 40)",
)
parser.add_argument(
    "--win_size",
    dest="win_size",
    action="store",
    default=0.025,
    help="window size during feature extraction (Default : 0.025 [25ms])",
)
parser.add_argument(
    "--norm_x",
    dest="norm_x",
    action="store",
    default=False,
    help="Normalize features s.t. mean = 0 std = 1",
)


def main(args):
    root = args.root
    target_path = root + "/processed/"
    train_path = ["train.tsv"]
    dev_path = ["dev.tsv"]
    test_path = ["test.tsv"]
    n_jobs = int(args.n_jobs)
    n_filters = args.n_filters
    win_size = args.win_size
    norm_x = args.norm_x

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    print("----------Processing Datasets----------")
    print("Training sets :", train_path)
    print("Validation sets :", dev_path)
    print("Test clean set :", test_path)

    # # mp32wav
    print("---------------------------------------")
    print("Processing mp32wav...", flush=True)

    print("Training", flush=True)
    tr_df = pd.read_csv(root + train_path[0], sep="\t")
    tr_df["sentence"] = tr_df["sentence"].apply(
        lambda val: unicodedata.normalize("NFKD", val).encode("ascii", "ignore").decode()
    )
    tr_df["path"] = root + "clips/" + tr_df["path"]
    tr_file_list = tr_df["path"].values
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(mp32wav)(i) for i in tqdm(tr_file_list)
    )
    print("Testing clean", flush=True)
    test_df = pd.read_csv(root + dev_path[0], sep="\t")
    test_df["sentence"] = test_df["sentence"].apply(
        lambda val: unicodedata.normalize("NFKD", val).encode("ascii", "ignore").decode()
    )
    test_df["path"] = root + "clips/" + test_df["path"]
    test_file_list = test_df["path"].values
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(mp32wav)(i) for i in tqdm(test_file_list)
    )

    # # wav 2 log-mel fbank
    print("---------------------------------------")
    print("Processing wav2logfbank...", flush=True)

    print("Training", flush=True)
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(wav2logfbank)(i[:-3] + "wav", win_size, n_filters, nfft=2048)
        for i in tqdm(tr_file_list)
    )

    print("Test clean", flush=True)
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(wav2logfbank)(i[:-3] + "wav", win_size, n_filters, nfft=2048)
        for i in tqdm(test_file_list)
    )

    # # log-mel fbank 2 feature
    print("---------------------------------------")
    print("Preparing Training Dataset...", flush=True)

    tr_file_list = tr_df["path"].str.replace("mp3", "fb" + str(n_filters) + ".npy").values
    tr_text = tr_df["sentence"].str.lower().replace("[^a-zA-Z0-9 ]", "", regex=True).values

    # Create char mapping
    char_map = char_mapping(tr_text, target_path)
    # text to index sequence
    tmp_list = []
    for text in tr_text:
        tmp = []
        for char in text:
            tmp.append(char_map[char])
        tmp_list.append(tmp)
    tr_text = tmp_list
    del tmp_list

    # write dataset
    file_name = "train.csv"

    print("Writing dataset to " + target_path + file_name + "...", flush=True)

    with open(target_path + file_name, "w") as f:
        f.write("idx,input,label\n")
        for i in range(len(tr_file_list)):
            f.write(str(i) + ",")
            f.write(tr_file_list[i] + ",")
            for char in tr_text[i]:
                f.write(" " + str(char))
            f.write("\n")

    print()
    print("Preparing Test clean Dataset...", flush=True)

    test_file_list = test_df["path"].str.replace("mp3", "fb" + str(n_filters) + ".npy").values
    tt_text = test_df["sentence"].str.lower().replace("[^a-zA-Z0-9 ]", "", regex=True).values

    # text to index sequence
    tmp_list = []
    for text in tt_text:
        tmp = []
        for char in text:
            tmp.append(char_map[char])
        tmp_list.append(tmp)
    tt_text = tmp_list
    del tmp_list

    # write dataset
    file_name = "test-clean.csv"

    print("Writing dataset to " + target_path + file_name + "...", flush=True)

    with open(target_path + file_name, "w") as f:
        f.write("idx,input,label\n")
        for i in range(len(test_file_list)):
            f.write(str(i) + ",")
            f.write(test_file_list[i] + ",")
            for char in tt_text[i]:
                f.write(" " + str(char))
            f.write("\n")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
