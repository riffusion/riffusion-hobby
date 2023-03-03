
import json
import pandas as pd
import sys
import os

dataset = pd.read_csv("../audiocaps/dataset/train.csv")

with open("../audiocaps_train/metadata.jsonl", 'w') as f:
    for _, row in dataset.iterrows():

        file_name = row['youtube_id'] + '.wav'
        full_path = "../audiocaps_train/" + file_name

        if not os.path.isfile(full_path):
            continue

        jsonline = {
            "file_name": file_name,
            "caption": row['caption'],
        }

        f.write( json.dumps(jsonline) + "\n" )

