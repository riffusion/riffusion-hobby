
import json
import pandas as pd
import sys
import os
from tqdm.auto import tqdm

from multiprocessing import Pool
import time
import os

import riffusion.cli as riffusioncli

dataset = pd.read_csv("../audiocaps/dataset/train.csv")

spectrogram_dir = '../audiocaps_train_spectrograms/'
if not os.path.isdir(spectrogram_dir):
    os.mkdir(spectrogram_dir)

with open("../audiocaps_train/metadata.jsonl", 'w') as f:
    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc='prepare metadata'):

        file_name = row['youtube_id'] + '.wav'
        full_path_audio = "../audiocaps_train/" + file_name

        if not os.path.isfile(full_path_audio):
            continue

        jsonline = {
            "file_name": file_name,
            "text": row['caption'],
        }

        f.write( json.dumps(jsonline) + "\n" )

print("metadata file is prepared")

# prepare spectrogram

print("prepare spectrograms")

def process_dataset_idx(i):
    row = dataset.iloc[i]

    file_name = row['youtube_id'] + '.wav'
    spectrogram_file_name = row['youtube_id'] + '.jpg'
    full_path_audio = "../audiocaps_train/" + file_name

    if not os.path.isfile(full_path_audio):
        return

    fill_path_spectrogram = spectrogram_dir + spectrogram_file_name

    riffusioncli.audio_to_image(audio=full_path_audio, image=fill_path_spectrogram)

    return


with Pool(processes=4) as pool:

    result = list(tqdm(pool.imap(process_dataset_idx, range(len(dataset))), total=len(dataset), desc='prepare spectrograms'))

# todo prepare spectrogram dataset with cli
# ~/anaconda3_new/envs/riffusion/bin/python3.9 -m riffusion.cli audio-to-image --audio ../audiocaps_train/000AjsqXq54.wav --image ./000AjsqXq54.jpg
