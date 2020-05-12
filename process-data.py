import glob, os
import enlighten
from multiprocessing import Pool
from utils import convert_to_wav
pool = Pool(8)
filelist=[f for f in os.listdir("data/mozilla/mp3") if f.endswith(".mp3")]
wavs=os.listdir("data/mozilla/wav")
filelist=set(filelist)
wavs=set(wavs)
filelist=list(filelist-wavs)
pbar = enlighten.Counter(total=len(filelist), desc='Basic', unit='ticks')
for i, _ in enumerate(pool.imap(convert_to_wav, filelist)):
    pbar.update()