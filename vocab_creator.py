import pandas as pd
import re
import unicodedata

filelist=["data/mozilla/train.tsv","data/mozilla/test.tsv","data/mozilla/validated.tsv","data/mozilla/invalidated.tsv","data/mozilla/dev.tsv"]

class vocabulary():
    def __init__(self,defaults=['<PAD>','<UNK>','<SOS>','<EOS>']):
        self.dict={}
        self.word_count={}
        self.counter = 0
        for d in defaults:
            self.add_word(d)
    def add_word(self,word):
        if word not in self.dict:
            self.dict[word] = self.counter
            self.counter+=1
        self.word_count[word] = self.word_count.get(word,0) +1
    def to_file(self,path):
        with open(path+"word_count.tsv", 'w') as count_file:
            with open(path+"vocabulary.tsv", 'w') as vocab_file:
                for word in self.dict:
                    vocab_file.write(word+'\t'+str(self.dict[word])+'\n')
                    count_file.write(word+'\t'+str(self.word_count[word])+'\n')


def clean_sentence(sentence):
    #print(sentence)
    sentence = unicodedata.normalize('NFKD', sentence).encode('ascii','ignore').decode("utf-8") #From unicode to ascii
    sentence = re.sub("[^\w\d'\s]+",'',sentence).lower().strip() #Remove every punctuation except '
    return sentence

vocab = vocabulary()
for file in filelist:
    print(f"Processing file {file}")
    tsv = pd.read_csv(file, sep='\t',low_memory=False)
    tsv.dropna(subset = ["sentence"], inplace=True)
    tsv['sentence'] = tsv['sentence'].apply(clean_sentence)

    for sentence in tsv['sentence'].values:
        sentence=re.sub(' +', ' ',sentence)
        for word in sentence.split(" "):
            for character in word:
                vocab.add_word(character)
    filename = "data/mozilla/processed/"+file.split("/")[-1].split(".")[0]+"_processed.tsv"

    #Fix file paths
    tsv['path'] = 'data/mozilla/mp3/' + tsv['path'].astype(str)
    tsv.to_csv(filename,sep='\t',index=False)
vocab.to_file('data/mozilla/')

