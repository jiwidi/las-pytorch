import pandas as pd
import re
import unicodedata
import os
import pdb
import argparse
import enlighten

libri_path = [
    "data/LibriSpeech/dev-clean/",
    "data/LibriSpeech/test-clean/",
    "data/LibriSpeech/train-clean-100/",
]

parser = argparse.ArgumentParser(description="Training script for LAS on Librispeech .")
parser.add_argument(
    "--check",
    metavar="config_path",
    type=bool,
    default=False,
    help="Activates check to see if torch can open all the files",
)
args = parser.parse_args()


class vocabulary:
    def __init__(self, defaults=["<SOS>", "<EOS>", "<PAD>", "<UNK>"]):
        self.dict = {}
        self.word_count = {}
        self.counter = 0
        for d in defaults:
            self.add_word(d)

    def add_word(self, word):
        if word not in self.dict:
            self.dict[word] = self.counter
            self.counter += 1
        self.word_count[word] = self.word_count.get(word, 0) + 1

    def to_file(self, path):
        with open(path + "word_count.tsv", "w") as count_file:
            with open(path + "vocabulary.tsv", "w") as vocab_file:
                for word in self.dict:
                    vocab_file.write(word + "\t" + str(self.dict[word]) + "\n")
                    count_file.write(word + "\t" + str(self.word_count[word]) + "\n")


def clean_sentence(sentence):
    # print(sentence)
    sentence = (
        unicodedata.normalize("NFKD", sentence).encode("ascii", "ignore").decode("utf-8")
    )  # From unicode to ascii
    sentence = (
        re.sub("[^\w\d'\s]+", "", sentence).lower().strip()
    )  # Remove every punctuation except '
    return sentence


vocab = vocabulary()
if not os.path.exists("data/LibriSpeech/processed/"):
    os.makedirs("data/LibriSpeech/processed/")
for folder in libri_path:
    users = os.listdir(folder)
    filepaths = []
    labels = []
    list_ids = []
    ids = 0
    for user in users:
        reads = os.listdir(folder + user)
        for subread in reads:
            filelist = os.listdir(folder + user + "/" + subread)
            transcript = [s for s in filelist if "trans.txt" in s][0]
            prepath = folder + user + "/" + subread + "/"
            with open(prepath + transcript, "r", encoding="utf-8") as t:
                for line in t:
                    parts = line.strip().split(" ", 1)
                    id = parts[0]
                    sentence = parts[1]
                    # Clean sentence
                    sentence = re.sub(" +", " ", sentence)
                    sentence = clean_sentence(sentence)
                    for word in sentence.split(" "):
                        for character in word:
                            vocab.add_word(character)
                    filepath = prepath + id + ".flac"
                    filepaths.append(filepath)
                    labels.append(sentence)
                    list_ids.append(ids)
                    ids += 1
    tsv = pd.DataFrame({"id": list_ids, "path": filepaths, "sentence": labels})
    filename = "data/LibriSpeech/processed/" + folder.split("/")[-2] + "_processed.tsv"

    if args.check:
        import torchaudio as ta

        pbar = enlighten.Counter(
            total=len(tsv), desc=f"Processing checks for {folder}", unit="ticks"
        )
        succed = []
        fail = []
        for filepath in tsv["path"]:
            try:
                wavform, sample_frequency = ta.load(filepath)
                succed.append(filepath)
            except:
                fail.append(filepath)
            pbar.update()
        print("")
        print(f"{len(succed)} Succeded | {len(fail)} Failed")
    tsv.to_csv(filename, sep="\t", index=False)


vocab.to_file("data/LibriSpeech/")
