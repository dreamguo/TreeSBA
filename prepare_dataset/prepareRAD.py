#!/usr/bin/env python
# coding=utf-8

import os
import sys
import ipdb
import torch
import argparse
import numpy as np

sys.path.append('.')
sys.path.append('./GenerativeLEGO/')

# define brick type
from utils import def_brick_type
def_brick_type.BRICK_TYPE = '0'

from utils import def_tree
from utils_RAD import buildLEGO, LEGO2ldr, ldr2action, action2voxel, action2obj
from utils_RAD.buildLEGO import My_Bricks

# Some code are modified from https://github.com/POSTECH-CVLab/Combinatorial-3D-Shape-Generation/


def work(opt):
    # 1. Randomly generate LEGO.npy.
    #       dataset/LEGO
    buildLEGO.make_LEGO(opt.num_train, opt.num_min, opt.num_max, opt.str_type,
                        opt.lego_dir, opt.object_type)

    # 2. Convert LEGO.npy to ldr file.
    #       dataset/ldr
    LEGO2ldr.lego2ldr(os.path.join(opt.lego_dir, opt.object_type), opt.ldr_dir, 
                      opt.str_type)

    # 2.5. (optional) remove the graph_dat if want to regenerate the graph
    # os.system('rm {}'.format(os.path.join(ROOT_PATH, 'graph_dat', opt.object_type + '.dat')))

    # 3. Extract actions from ldr file and also save graph.dat files.
    #       dataset/graph_dat
    #       dataset/tree_actions
    #       dataset/tree_dep_actions
    ldr2action.ldr2action(opt)

    # 4. Construct 3D models from actions and project them to images.
    #       dataset/voxel_img
    action2voxel.action2voxel_img(opt.action_dir, opt.voxel_img_dir, opt.object_type,
                                  opt.voxel_size, opt.pad_num, opt.brick_num)

    # 5. (optional) Construct 3D models from actions and save them.
    #       dataset/voxel
    # action2voxel.action2voxel(opt.action_dir, opt.voxel_dir, opt.object_type,
    #                           opt.voxel_size, opt.pad_num, opt.brick_num)

    # 6. (optional) Obtain obj file from assembly action sequence. 
    #       dataset/obj
    # action2obj.action2obj(opt)


if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if cuda else "cpu")

    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.str_type = def_brick_type.BRICK_TYPE
    opt.pad_num = 999
    opt.device = device
    opt.voxel_size = 64
    # opt.connect_num = def_tree.CONNECT_TYPE
    ROOT_PATH = os.path.join(os.getcwd(), 'dataset')
    opt.ldr_dir = os.path.join(ROOT_PATH, 'ldr')
    opt.obj_dir = os.path.join(ROOT_PATH, 'obj')
    opt.lego_dir = os.path.join(ROOT_PATH, 'LEGO')
    opt.voxel_dir = os.path.join(ROOT_PATH, 'voxel')
    opt.voxel_img_dir = os.path.join(ROOT_PATH, 'voxel_img')
    opt.action_dir = os.path.join(ROOT_PATH, 'tree_actions')
    opt.root_path = os.getcwd()

    # Change if needed
    opt.num_min = 15
    opt.num_max = 50
    opt.num_train = np.arange(10)
    opt.brick_num = opt.num_max + 1
    if opt.str_type == '1':
        opt.object_type = 'random{}-{}to{}'.format(opt.str_type, opt.num_min, opt.num_max)
    elif opt.str_type == '0':
        opt.object_type = 'random{}to{}'.format(opt.num_min, opt.num_max)
    if 0:
        opt.lego_dir = os.path.join(opt.lego_dir, 'label')
        assert opt.str_type == '0'

        for type_ in ['label21', 'label22', 'label25', 'label26']:
            opt.object_type = type_
            work(opt)
        exit()

    work(opt)

'''
    STR_LABEL_PARALLEL = 'label01'
    STR_LABEL_PERPENDICULAR = 'label02'

    STR_LABEL_BAR = 'label11'
    STR_LABEL_LINE = 'label12'
    STR_LABEL_PLATE = 'label13'
    STR_LABEL_WALL = 'label14'
    STR_LABEL_CUBOID = 'label15'
    STR_LABEL_SQUAREPYRAMID = 'label16'

    STR_LABEL_BENCH = 'label21'
    STR_LABEL_SOFA = 'label22'
    STR_LABEL_CUP = 'label23'
    STR_LABEL_HOLLOW = 'label24'
    STR_LABEL_TABLE = 'label25'
    STR_LABEL_CAR = 'label26'

    STR_LABEL_RANDOM = 'random'
'''
