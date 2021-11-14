import math
import os
import time
import pickle
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import ast

from opt import *

def resize_bbox(row,hight,width):
    bbox=row["nao_bbox"]
    new_bbox= [bbox[0]/hight*456,bbox[1]/width*256,bbox[2]/hight*456,bbox[3]/width*256]

    return [round(coord) for coord in new_bbox]


# given each video's resolution and fps
# 1.resize the bounding box to 456x256 (size of donwloader RGB frames)
# 2.add frames before each train/test sample according to sample_time_length (how long before the sample frame) and
#   sample_fps ( how many frames in each sampled second)
def add_previous_frames(sample_time_length=5,sample_fps=3):
    video_info_path = os.path.join(args.data_path, 'EPIC_video_info.csv')
    video_info = pd.read_csv(video_info_path)
    all_par_video_id = sorted(id)
    for par_video_id in all_par_video_id:
        video_id = par_video_id[3:]
        anno_file_path = os.path.join(args.data_path, annos_path, f'nao_{par_video_id}.csv')
        if os.path.exists(anno_file_path):
            print(f'current video id: {video_id}')
            current_video_ino = video_info.loc[video_info['video'] == video_id]
            fps = current_video_ino.iloc[0]["fps"]
            resolution = current_video_ino.iloc[0]["resolution"]
            height = int(resolution[0:4])
            width = int(resolution[5:])
            sample_steps = fps // sample_fps
            previous_frames_helper = [-sample_steps * timestep for timestep in
                                      range(sample_fps * sample_time_length, 0, -1)]
            annotations = pd.read_csv(anno_file_path, converters={"nao_bbox": literal_eval})
            if annotations.shape[0] > 0:
                annotations['participant_id'] = annotations.apply(lambda row: row["id"][0:3], axis=1)
                annotations['video_id'] = annotations.apply(lambda row: row["id"][3:], axis=1)
                annotations['fps'] = fps
                # print(previous_frames_helper)
                annotations['previous_frames'] = annotations.apply(
                    lambda row: ([(row['frame'] + i) for i in previous_frames_helper]), axis=1)
                annotations['previous_frames'] = annotations.apply(
                    lambda row: [frame if frame > 0 else 1 for frame in row['previous_frames']], axis=1)
                annotations['nao_bbox_resized'] = annotations.apply(resize_bbox, args=[height, width], axis=1)
                annotations.to_csv(anno_file_path, index=False)


if __name__=='__main__':
    add_previous_frames(sample_time_length=5,sample_fps=3)