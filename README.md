## LAS-Pytorch

This is my pytorch implementation for the [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211v2) (LAS) google ASR deep learning model. I used both the mozilla [Common voice](https://voice.mozilla.org/en/datasets) dataset and the [LibriSpeech](https://www.openslr.org/12) dataset.

![LAS Network architecture](img/las.png)

The feature transformation is done on the fly while loading the files thanks to torchaudio.

## Results

## How to run it

### Requirements
Code is setup to run with both the mozilla [Common voice](https://voice.mozilla.org/en/datasets) dataset and the [LibriSpeech](https://www.openslr.org/12) dataset. If you want to run the code you should download the datasets and extract them under data/ to follow the following structure:

### Data
```
data
├── LibriSpeech
│   ├── BOOKS.TXT
│   ├── CHAPTERS.TXT
│   ├── dev-clean/
│   ├── LICENSE.TXT
│   ├── README.TXT
│   ├── SPEAKERS.TXT
│   ├── test-clean/
│   ├── train-clean-100/
└── mozilla
    ├── dev.tsv
    ├── invalidated.tsv
    ├── mp3/
    ├── other.tsv
    ├── test.tsv
    ├── train.tsv
    ├── validated.tsv
```
And run the following commands to process and collect all files.

```
$ python prepare_data-libri.py
$ python prepare_data-common-voice.py
```
This will create a `processed/` folder inside each of the datasets folders along with vocabulary and word count files.

### Training
Execute the train script along with the yaml config file for the desired dataset.
```
$ python train.py --config_path config/librispeech-config.yaml
# Or
$ python train.py --config_path config/common_voice-config.yaml
```

Loss and lert will be logged to the `runs/` folder, you can check them by running tensoboard in the root directory.
