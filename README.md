## LAS-Pytorch

This is my pytorch implementation for the [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211v2) (LAS) google ASR deep learning model. I used both the mozilla [Common voice](https://voice.mozilla.org/en/datasets) dataset and the [LibriSpeech](https://www.openslr.org/12) dataset.

![LAS Network architecture](img/las.png)

The feature transformation is done on the fly while loading the files thanks to torchaudio.

## Results

Still training lel

## How to run it

### Requirements
Code is setup to run with both the mozilla [Common voice](https://voice.mozilla.org/en/datasets) dataset and the [LibriSpeech](https://www.openslr.org/12) dataset. If you want to run the code you should download the datasets and extract them under data/ or run the script `utils/download_data.py` which will download it and extract it in the following format:

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
│   └── train-clean-100/
└── mozilla
    ├── dev.tsv
    ├── invalidated.tsv
    ├── mp3/
    ├── other.tsv
    ├── test.tsv
    ├── train.tsv
    └──  validated.tsv
```

So run
```

#Remove flags if you want to avoid download that specific dataset
$ python utils/download_data.py --libri --common
```

And run the following commands to process and collect all files.

```
#Still in utils/
$ python utils/prepare_librispeech.py --root $ABSOLUTEPATH TO DATASET
$ python uitls/prepare_common-voice.py --root $ABSOLUTEPATH TO DATASET
```
This will create a `processed/` folder inside each of the datassets containing the csvs with teh data neccesary to train along vocabulary and word count files.

### Training
Execute the train script along with the yaml config file for the desired dataset.
```
$ python train.py --config_path config/librispeech-config.yaml
# Or
$ python train.py --config_path config/common_voice-config.yaml
```

Loss and lert will be logged to the `runs/` folder, you can check them by running tensoboard in the root directory.
