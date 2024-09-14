import os
import ipdb
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F

from utils import render


def vt2obj(t_verts_l, t_trians_l, path, obj_name):
    obj_file = open(os.path.join(path, obj_name), 'w')
    t_verts = t_verts_l[0].detach().cpu().numpy()
    # +1 because obj triangles start from 1 while t_triangles start from 0
    t_trians = t_trians_l[0].detach().cpu().numpy() + 1
    obj_content = ''
    for item in t_verts:
        obj_content += "v {0} {1} {2}\n".format(item[0],item[1],item[2])
    for item in t_trians:
        obj_content += "f {0} {1} {2}\n".format(item[0],item[1],item[2])
    obj_file.write(obj_content)
    obj_file.close()


def action2obj(opt):
    obj_path = os.path.join(opt.obj_dir, opt.object_type)
    action_path = os.path.join(opt.action_dir, opt.object_type)
    if not os.path.exists(obj_path):
        os.makedirs(obj_path)
        print("making ", obj_path)
    for obj_f in tqdm(sorted(os.listdir(action_path))):
        # TODO: not suitable for different root
        if int(obj_f.split('.')[0].split('_')[2][4:]) != 0:
            continue
        action = np.load(os.path.join(action_path, obj_f))
        action = torch.from_numpy(action.reshape(1, 2, -1)).to(opt.device)
        verts, trians = render.seq2vt(action, opt)
        obj_name = '_'.join(obj_f.split('_')[:2]) + '.obj'
        vt2obj(verts, trians, obj_path, obj_name)


def self_act2obj(opt):
    action = np.array([[31, 25, 0, 0], [0, 0, 6, 0]])
    action = torch.from_numpy(action.reshape(1, 2, -1)).to(opt.device)
    action = F.pad(action, (0, opt.brick_num - action.shape[2]))
    verts, trians = render.seq2vt(action, opt)
    vt2obj(verts, trians, opt.obj_dir, 'test.obj')
