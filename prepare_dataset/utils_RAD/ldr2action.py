import ipdb
import argparse

from utils import def_tree, helpers


def ldr2action(opt):
    helpers.GetDGMGDataset(config=vars(opt))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.pad_num = 999
    opt.brick_num = 16
    opt.voxel_size = 64
    opt.object_type = 'random10to15'
    # opt.connect_num = def_tree.CONNECT_TYPE
    ldr2action(opt)
