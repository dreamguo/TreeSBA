import os
import sys
import ipdb
import numpy as np
import argparse
from geometric_primitives import utils_io, utils_meshes

sys.path.append('.')
from utils import def_brick_type
def_brick_type.BRICK_TYPE = '1'


def work(in_filename):
    my_bricks = np.load(in_filename, allow_pickle = True)
    mesh_bricks, mesh_cubes = utils_meshes.get_mesh_bricks(my_bricks[()])
    utils_io.visualize(mesh_bricks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=1)
    opt = parser.parse_args()
    ROOT_PATH = 'dataset/LEGO'
    opt.str_type = def_brick_type.BRICK_TYPE

    # Change if needed
    opt.num_min = 10
    opt.num_max = 50
    opt.brick_num = opt.num_max + 1
    opt.object_type = 'random{}-{}to{}'.format(opt.str_type, opt.num_min, opt.num_max)

    dir_ = os.path.join(ROOT_PATH, opt.object_type)
    in_filename = os.path.join(dir_, '{}_{:06d}.npy'.format(opt.object_type, opt.id))
    ipdb.set_trace()
    work(in_filename)
