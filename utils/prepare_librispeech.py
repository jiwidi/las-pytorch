# This preprocess scripts was made by https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch, I just edited some stuff
from pydub import AudioSegment
import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import scipy.io.wavfile as wav
from python_speech_features import logfbank
from functions import traverse, wav2logfbank, flac2wav, norm, char_mapping
import argparse


parser = argparse.ArgumentParser(description="Librispeech preprocess.")

parser.add_argument(
    "--root",
    metavar="root",
    type=str,
    required=True,
    help="Absolute file path to LibriSpeech. (e.g. /usr/downloads/LibriSpeech/)",
)

parser.add_argument(
    "--n_jobs",
    dest="n_jobs",
    action="store",
    default=-2,
    help="number of cpu availible for preprocessing.\n -1: use all cpu, -2: use all cpu but one",
)
parser.add_argument(
    "--n_filters", dest="n_filters", action="store", default=40, help="number of filters for fbank. (Default : 40)",
)
parser.add_argument(
    "--win_size",
    dest="win_size",
    action="store",
    default=0.025,
    help="window size during feature extraction (Default : 0.025 [25ms])",
)
parser.add_argument(
    "--norm_x", dest="norm_x", action="store", default=False, help="Normalize features s.t. mean = 0 std = 1",
)


def main(args):
    root = args.root
    target_path = root + "/processed/"
    train_path = ["train-clean-100/" , "train-clean-360/", "train-other-500/"]
    dev_path = ["dev-clean/", "dev-other/"]
    test_clean_path = ["test-clean/"]
    test_other_path = ["test-other/"]
    n_jobs = args.n_jobs
    n_filters = args.n_filters
    win_size = args.win_size
    norm_x = args.norm_x

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    print("----------Processing Datasets----------")
    print("Training sets :", train_path)
    print("Validation sets :", dev_path)
    print("Test clean set :", test_clean_path)
    print("Test other set :", test_other_path)

    # # flac2wav
    print("---------------------------------------")
    print("Processing flac2wav...", flush=True)

    print("Training", flush=True)
    tr_file_list = traverse(root, train_path)
    results = Parallel(n_jobs=n_jobs, backend="threading")(delayed(flac2wav)(i) for i in tqdm(tr_file_list))

    print("Validation", flush=True)
    dev_file_list = traverse(root, dev_path)
    results = Parallel(n_jobs=n_jobs, backend="threading")(delayed(flac2wav)(i) for i in tqdm(dev_file_list))

    print("Testing clean", flush=True)
    tt_clean_file_list = traverse(root, test_clean_path)
    results = Parallel(n_jobs=n_jobs, backend="threading")(delayed(flac2wav)(i) for i in tqdm(tt_clean_file_list))

    print("Testing other", flush=True)
    tt_other_file_list = traverse(root, test_other_path)
    results = Parallel(n_jobs=n_jobs, backend="threading")(delayed(flac2wav)(i) for i in tqdm(tt_other_file_list))

    # # wav 2 log-mel fbank
    print("---------------------------------------")
    print("Processing wav2logfbank...", flush=True)

    print("Training", flush=True)
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(wav2logfbank)(i[:-4] + "wav", win_size, n_filters) for i in tqdm(tr_file_list)
    )

    print("Validation", flush=True)
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(wav2logfbank)(i[:-4] + "wav", win_size, n_filters) for i in tqdm(dev_file_list)
    )

    print("Test clean", flush=True)
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(wav2logfbank)(i[:-4] + "wav", win_size, n_filters) for i in tqdm(tt_clean_file_list)
    )

    print("Test other", flush=True)
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(wav2logfbank)(i[:-4] + "wav", win_size, n_filters) for i in tqdm(tt_other_file_list)
    )

    # # log-mel fbank 2 feature
    print("---------------------------------------")
    print("Preparing Training Dataset...", flush=True)

    tr_file_list = traverse(root, train_path, search_fix=".fb" + str(n_filters))
    tr_text = traverse(root, train_path, return_label=True)

    # X = []
    # for f in tr_file_list:
    #     X.append(np.load(f))

    # # Normalize X
    # if norm_x:
    #     mean_x = np.mean(np.concatenate(X, axis=0), axis=0)
    #     std_x = np.std(np.concatenate(X, axis=0), axis=0)

    #     results = Parallel(n_jobs=n_jobs, backend="threading")(
    #         delayed(norm)(i, mean_x, std_x) for i in tqdm(tr_file_list)
    #     )

    # Sort data by signal length (long to short)
    # audio_len = [len(x) for x in X]

    # tr_file_list = [tr_file_list[idx] for idx in reversed(np.argsort(audio_len))]
    # tr_text = [tr_text[idx] for idx in reversed(np.argsort(audio_len))]

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
    print("Preparing Validation Dataset...", flush=True)

    dev_file_list = traverse(root, dev_path, search_fix=".fb" + str(n_filters))
    dev_text = traverse(root, dev_path, return_label=True)

    # X = []
    # for f in dev_file_list:
    #     X.append(np.load(f))

    # # Normalize X
    # if norm_x:
    #     results = Parallel(n_jobs=n_jobs, backend="threading")(
    #         delayed(norm)(i, mean_x, std_x) for i in tqdm(dev_file_list)
    #     )

    # # Sort data by signal length (long to short)
    # audio_len = [len(x) for x in X]

    # dev_file_list = [dev_file_list[idx] for idx in reversed(np.argsort(audio_len))]
    # dev_text = [dev_text[idx] for idx in reversed(np.argsort(audio_len))]

    # text to index sequence
    tmp_list = []
    for text in dev_text:
        tmp = []
        for char in text:
            tmp.append(char_map[char])
        tmp_list.append(tmp)
    dev_text = tmp_list
    del tmp_list

    # write dataset
    file_name = "dev.csv"

    print("Writing dataset to " + target_path + file_name + "...", flush=True)

    with open(target_path + file_name, "w") as f:
        f.write("idx,input,label\n")
        for i in range(len(dev_file_list)):
            f.write(str(i) + ",")
            f.write(dev_file_list[i] + ",")
            for char in dev_text[i]:
                f.write(" " + str(char))
            f.write("\n")

    print()
    print("Preparing Test clean Dataset...", flush=True)

    test_file_list = traverse(root, test_clean_path, search_fix=".fb" + str(n_filters))
    tt_text = traverse(root, test_clean_path, return_label=True)

    # X = []
    # for f in test_file_list:
    #     X.append(np.load(f))

    # # Normalize X
    # if norm_x:
    #     results = Parallel(n_jobs=n_jobs, backend="threading")(
    #         delayed(norm)(i, mean_x, std_x) for i in tqdm(test_file_list)
    #     )

    # Sort data by signal length (long to short)
    # audio_len = [len(x) for x in X]

    # test_file_list = [test_file_list[idx] for idx in reversed(np.argsort(audio_len))]
    # tt_text = [tt_text[idx] for idx in reversed(np.argsort(audio_len))]

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

    print("Preparing Test other Dataset...", flush=True)

    test_file_list = traverse(root, test_other_path, search_fix=".fb" + str(n_filters))
    tt_text = traverse(root, test_other_path, return_label=True)

    # X = []
    # for f in test_file_list:
    #     X.append(np.load(f))

    # # Normalize X
    # if norm_x:
    #     results = Parallel(n_jobs=n_jobs, backend="threading")(
    #         delayed(norm)(i, mean_x, std_x) for i in tqdm(test_file_list)
    #     )

    # Sort data by signal length (long to short)
    # audio_len = [len(x) for x in X]

    # test_file_list = [test_file_list[idx] for idx in reversed(np.argsort(audio_len))]
    # tt_text = [tt_text[idx] for idx in reversed(np.argsort(audio_len))]

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
    file_name = "test-other.csv"

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
