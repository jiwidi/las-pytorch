from pydub import AudioSegment

##GLOBAL VARIABLES
IGNORE_ID = -1


def convert_to_wav(file):
    sound = AudioSegment.from_mp3(f"data/mozilla/mp3/{file}")
    sound.export(f"data/mozilla/wav/{file}", format="wav")

def pad_list(xs, pad_value):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad