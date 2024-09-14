#!/usr/bin/env python
# coding=utf-8
import os
import ipdb
import torch
import numpy as np
from torch.utils.data.dataset import Dataset

from utils import eval_utils


def voxel2img(voxel):
    imgs = []
    for i in range(3):
        img = voxel.any(axis=i)
        imgs.append(img.astype(np.float32))
    return np.array(imgs)

def voxel2noise_img(voxel, erase_portion):
    if erase_portion > 0:
        x, y, z = np.where(voxel==1)
        idxs = np.arange(len(x))
        np.random.shuffle(idxs)
        erase_num = int(erase_portion * len(x))
        erase_idxs = idxs[:erase_num]
        voxel[x[erase_idxs], y[erase_idxs], z[erase_idxs]] = 0
    return voxel2img(voxel)

def norm_voxel(voxel, pad_size, data):
    half_size = pad_size // 2
    img0 = voxel.any(axis=0)
    img1 = voxel.any(axis=1)
    img2 = voxel.any(axis=2)
    if data.split('_')[0] == 'mnist':
        # 'median' for act-train root != 0, 'min' for act-train root == 0
        root_z = int(np.min(np.where(img1==1)[1]))
        root_x = int(np.max(np.where(img1[:, root_z]==1)[0]))
        root_y = int(np.min(np.where((img2[root_x] & img0[:, root_z])==1)[0])) + 2
    elif data.split('_')[0] == 'modelnet':
        root_y = int(np.ceil(np.median(np.where(img0==1)[0])))
        root_z = int(np.median(np.where(img0[root_y]==1)[0]))
        root_x = int(np.ceil(np.median(np.where((img2[:, root_y] & img1[:, root_z])==1)[0])))
    x_left = half_size - root_x
    y_left = half_size - root_y
    z_left = half_size - root_z
    x_right = pad_size - x_left - voxel.shape[0]
    y_right = pad_size - y_left - voxel.shape[1]
    z_right = pad_size - z_left - voxel.shape[2]
    voxel = np.pad(voxel, ((x_left,x_right),(y_left,y_right),(z_left,z_right)))
    return voxel


MODELNET3_LIST = ['airplane', 'monitor', 'table']
MODELNET_SELECT_LIST = ['airplane', 'monitor', 'table', 'chair', 'desk','bench','piano',
                        'stool','toilet']
class TrainSet(Dataset):
    def __init__(self, opt):
        self.datas = []
        self.pad_num = opt.pad_num
        self.brick_num = opt.brick_num
        self.voxel_size = opt.voxel_size
        self.object_type = opt.object_type
        self.modelnet3 = opt.train_modelnet3
        self.erase_portion = opt.erase_portion
        self.root_aug_range = opt.root_aug_range
        self.act_path = opt.act_path + self.object_type
        self.voxels_path = opt.voxels_path + self.object_type
        self.dep_act_path = opt.dep_act_path + self.object_type
        self.voxel_img_path = opt.voxel_img_path + self.object_type

        if (self.object_type.split('_')[0] == 'mnist' or
            self.object_type.split('_')[0] == 'modelnet'):
            self.voxels_path = os.path.join(self.voxels_path, 'train')
            for data in sorted(os.listdir(self.voxels_path)):
                if (self.modelnet3 and data.split('_')[1] not in MODELNET3_LIST):
                    continue
                try:
                    voxel = np.load(os.path.join(self.voxels_path, data))
                    norm_voxel(voxel, self.voxel_size, data)
                    self.datas.append(data)
                except:
                    continue
        else:
            obj_paths = {}
            for data in sorted(os.listdir(self.voxel_img_path)):
                root_n = int(data.split('.')[0].split('_')[2][4:])
                obj_n = int(data.split('.')[0].split('_')[1])
                if obj_n in obj_paths.keys():
                    obj_paths[obj_n] = min(root_n, obj_paths[obj_n])
                else:
                    obj_paths[obj_n] = root_n
            for data in sorted(os.listdir(self.voxel_img_path)):
                root_n = int(data.split('.')[0].split('_')[2][4:])
                obj_n = int(data.split('.')[0].split('_')[1])
                if root_n != obj_paths[obj_n]:
                    continue
                if obj_n > opt.num_train:
                    continue
                self.datas.append(data)

    def __getitem__(self, index):
        voxel, imgs, data = self.load_voxel(self.datas[index])
        if (self.object_type.split('_')[0] == 'mnist' or 
            self.object_type.split('_')[0] == 'modelnet' or 
            self.object_type.split('_')[0][:5] == 'label'):
            return imgs, 0, 0, voxel, data.split('.')[0]
        action = np.load(os.path.join(self.act_path, data))[:, :self.brick_num]
        size =  ((0,0), (0, self.brick_num - len(action[0])))
        action = np.pad(action, size, 'constant', constant_values=self.pad_num)
        dep_action = np.load(os.path.join(self.dep_act_path, data))[:self.brick_num, :]
        size =  ((0, self.brick_num - dep_action.shape[0]), (0,0))
        dep_action = np.pad(dep_action, size, 'constant', constant_values=0)
        return imgs, action, dep_action, voxel, data.split('.')[0]

    def __len__(self):
        return len(self.datas)

    def aug_voxel(self, data):
        if self.root_aug_range != 1:
            while True:
                augment_idx = np.random.choice(self.root_aug_range)
                name = data.split('.')[0].split('_')
                name[2] = 'root' + str(augment_idx)
                data = '_'.join(name) + '.npy'
                if os.path.exists(os.path.join(self.voxel_img_path, data)):
                    break
        actions = np.load(os.path.join(self.act_path, data), allow_pickle=True)
        voxel = eval_utils.get_tree_voxel(torch.tensor(actions), self.voxel_size,
                                     self.pad_num, self.brick_num)
        return np.array(voxel), data

    def load_voxel(self, data):
        if (self.object_type.split('_')[0] == 'mnist' or 
            self.object_type.split('_')[0] == 'modelnet'):
            voxel = np.load(os.path.join(self.voxels_path, data))
            voxel = norm_voxel(voxel, self.voxel_size, data)
            imgs = voxel2img(voxel)
        elif self.object_type.split('_')[0][:5] == 'label':
            voxel = np.load(os.path.join(self.voxels_path, data))
            imgs = voxel2img(voxel)
        else:
            voxel, data = self.aug_voxel(data)
            imgs = voxel2noise_img(voxel, self.erase_portion)
        return voxel, imgs, data


class TestSet(Dataset):
    def __init__(self, opt):
        self.datas = []
        self.pad_num = opt.pad_num
        self.brick_num = opt.brick_num
        self.voxel_size = opt.voxel_size
        self.modelnet3 = opt.test_modelnet3
        self.test_object_type = opt.test_object_type
        self.act_path = opt.act_path + self.test_object_type
        self.voxels_path = opt.voxels_path + self.test_object_type
        self.dep_act_path = opt.dep_act_path + self.test_object_type
        self.voxel_img_path = opt.voxel_img_path + self.test_object_type

        if (self.test_object_type.split('_')[0] == 'mnist' or
            self.test_object_type.split('_')[0] == 'modelnet'):
            self.voxels_path = os.path.join(self.voxels_path, 'test')
            for data in sorted(os.listdir(self.voxels_path)):
                if (self.modelnet3 and data.split('_')[1] not in MODELNET3_LIST):
                    continue
                if data.split('_')[1] not in MODELNET_SELECT_LIST:
                    continue
                try:
                    voxel = np.load(os.path.join(self.voxels_path, data))
                    norm_voxel(voxel, self.voxel_size, data)
                    self.datas.append(data)
                except:
                    continue
        else:
            obj_paths = {}
            for data in sorted(os.listdir(self.voxel_img_path)):
                root_n = int(data.split('.')[0].split('_')[2][4:])
                obj_n = int(data.split('.')[0].split('_')[1])
                if obj_n in obj_paths.keys():
                    obj_paths[obj_n] = min(root_n, obj_paths[obj_n])
                else:
                    obj_paths[obj_n] = root_n
            for data in sorted(os.listdir(self.voxel_img_path)):
                root_n = int(data.split('.')[0].split('_')[2][4:])
                obj_n = int(data.split('.')[0].split('_')[1])
                if root_n != obj_paths[obj_n]:
                    continue
                if (obj_n <= opt.num_train or obj_n > opt.num_train + opt.num_test):
                    continue
                self.datas.append(data)

    def __getitem__(self, index):
        data = self.datas[index]
        voxel, imgs = self.load_voxel(data)
        if (self.test_object_type.split('_')[0] == 'mnist' or 
            self.test_object_type.split('_')[0] == 'modelnet' or 
            self.test_object_type.split('_')[0][:5] == 'label'):
            return imgs, 0, 0, voxel, data.split('.')[0]
        action = np.load(os.path.join(self.act_path, data))
        size =  ((0,0), (0, self.brick_num - len(action[0])))
        action = np.pad(action, size, 'constant', constant_values=self.pad_num)
        dep_action = np.load(os.path.join(self.dep_act_path, data))
        size =  ((0, self.brick_num - dep_action.shape[0]), (0,0))
        dep_action = np.pad(dep_action, size, 'constant', constant_values=0)
        return imgs, action, dep_action, voxel, data.split('.')[0]

    def __len__(self):
        return len(self.datas)

    def aug_voxel(self, data):
        actions = np.load(os.path.join(self.act_path, data), allow_pickle=True)
        voxel = eval_utils.get_tree_voxel(torch.tensor(actions), self.voxel_size,
                                     self.pad_num, self.brick_num)
        return np.array(voxel)

    def load_voxel(self, data):
        if (self.test_object_type.split('_')[0] == 'mnist' or 
            self.test_object_type.split('_')[0] == 'modelnet'):
            voxel = np.load(os.path.join(self.voxels_path, data))
            voxel = norm_voxel(voxel, self.voxel_size, data)
        elif self.test_object_type.split('_')[0][:5] == 'label':
            voxel = np.load(os.path.join(self.voxels_path, data))
        else:
            voxel = self.aug_voxel(data)
        imgs = voxel2img(voxel)
        return voxel, imgs
