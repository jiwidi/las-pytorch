###
# This script will download the datasets tar files and extract them in the data/ directory where all the other scripts for training
# or processing expect the data to be in
###


import os
import wget
import subprocess

mozilla_urls = {
    "common_voice": "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/en.tar.gz"
}
lib_urls = {
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "train-clean-100": "hhttps://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
}
if not os.path.exists("data/"):
    os.makedirs("data/")
if not os.path.exists("data/mozilla/"):
    os.makedirs("data/mozilla/")

print("Download librispeech dataset")
for name, url in lib_urls.items():
    output_directory = url.split("/")[-1]
    print(f"Downloading {output_directory}")
    filename = wget.download(url, out=output_directory)
    print(f"Extracting {output_directory}")
    subprocess.run(["tar", "-xf", filename, "-C", "data/"])
    subprocess.run(["rm", "-f", filename])
print("Done downloading librispeech dataset")


print("Downloading common voice dataset")
for name, url in mozilla_urls.items():
    output_directory = url.split("/")[-1]
    print(f"Downloading {output_directory}")
    filename = wget.download(url, out=output_directory)
    print(f"Extracting {output_directory}")
    subprocess.run(["tar", "-xf", filename, "-C", "data/mozilla/"])
    subprocess.run(["rm", "-f", filename])
print("Done downloading common voice dataset")
print("Downloading of datasets complete, run now the prepare data scripts for each one")
