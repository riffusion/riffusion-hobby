
import json
import pandas as pd
import sys
import os
from tqdm.auto import tqdm

from multiprocessing import Pool
import time
import os

import argparse
import riffusion.cli as riffusioncli

# riffusion repo must be script working directory

parser = argparse.ArgumentParser(description="Dataset preparation script.")
parser.add_argument("--prepare-metadata", default=False, action='store_true')
parser.add_argument("--prepare-spectrograms", default=False, action='store_true')

args = parser.parse_args()

dataset = pd.read_csv("../audiocaps/dataset/train.csv")

audio_dir = "../audiocaps_train/"
spectrogram_dir = '../audiocaps_train_spectrograms/'
if not os.path.isdir(spectrogram_dir):
    os.mkdir(spectrogram_dir)

# prepare spectrogram

def process_dataset_idx(i):
    try:
        row = dataset.iloc[i]

        audio_files = set(os.listdir(audio_dir))
        spectrogram_files = set(os.listdir(spectrogram_dir))

        file_name = row['youtube_id'] + '.wav'
        spectrogram_file_name = row['youtube_id'] + '.jpg'
        full_path_audio = audio_dir + file_name

        if spectrogram_file_name in spectrogram_files:
            return

        if file_name not in audio_files:
            return

        fill_path_spectrogram = spectrogram_dir + spectrogram_file_name

        riffusioncli.audio_to_image(audio=full_path_audio, image=fill_path_spectrogram)
    except Exception as e:
        print("cant process", i, "exception", e)

    return

if args.prepare_spectrograms:
    print("prepare spectrograms")

    with Pool(processes=4) as pool:

        result = list(tqdm(pool.imap(process_dataset_idx, range(len(dataset))), total=len(dataset), desc='prepare spectrograms'))

    print("sprctrogram files are prepared:", spectrogram_dir)

# to prepare spectrogram dataset with cli
# ~/anaconda3_new/envs/riffusion/bin/python3.9 -m riffusion.cli audio-to-image --audio ../audiocaps_train/000AjsqXq54.wav --image ./000AjsqXq54.jpg

if args.prepare_metadata:
    full_metadata_file_path = spectrogram_dir + "metadata.jsonl"
    with open(full_metadata_file_path, 'w') as f:
        for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc='prepare metadata'):

            file_name = row['youtube_id'] + '.wav'
            spectrogram_file_name = row['youtube_id'] + '.jpg'
            full_path_audio = "../audiocaps_train/" + file_name
            full_path_spectrogram = spectrogram_dir + spectrogram_file_name

            if not os.path.isfile(full_path_audio):
                continue

            if not os.path.isfile(full_path_spectrogram):
                continue

            jsonline = {
                "file_name": full_path_spectrogram,
                "text": row['caption'],
            }

            f.write( json.dumps(jsonline) + "\n" )

    print("metadata file is prepared", full_metadata_file_path)
