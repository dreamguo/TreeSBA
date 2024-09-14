import os
import sys
import ipdb
import torch
import argparse
import numpy as np
from tqdm import tqdm

from utils import eval_utils


def voxel2img(voxel):
    imgs = []
    for i in range(3):
        img = voxel.any(axis=i)
        imgs.append(np.array(img.float()))
    return np.array(imgs)

def action2voxel_img(action_dir, voxel_img_dir, object_type, voxel_size, pad_num, brick_num):
    action_path = os.path.join(action_dir, object_type)
    voxel_img_path = os.path.join(voxel_img_dir, object_type)
    if not os.path.exists(voxel_img_path):
        os.makedirs(voxel_img_path)
    for filename in tqdm(sorted(os.listdir(action_path))):
        actions = np.load(os.path.join(action_path, filename), allow_pickle=True)
        voxel = eval_utils.get_tree_voxel(torch.tensor(actions), voxel_size, pad_num, brick_num)
        np.save(os.path.join(voxel_img_path, filename), voxel2img(voxel))

def action2voxel(action_dir, voxel_dir, object_type, voxel_size, pad_num, brick_num):
    action_path = os.path.join(action_dir, object_type)
    voxel_path = os.path.join(voxel_dir, object_type)
    if not os.path.exists(voxel_path):
        os.makedirs(voxel_path)
    for filename in tqdm(sorted(os.listdir(action_path))):
        if int(filename.split('.')[0].split('_')[2][4:]) != 0:
            continue
        actions = np.load(os.path.join(action_path, filename), allow_pickle=True)
        voxel = eval_utils.get_tree_voxel(torch.tensor(actions), voxel_size, pad_num, brick_num)
        np.save(os.path.join(voxel_path, filename), voxel)
