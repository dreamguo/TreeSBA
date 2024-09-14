import os
import ipdb
import configargparse

from utils import def_tree, def_brick_type


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True)
    parser.add_argument("--use_wandb", type=int, default=0)

    # data
    parser.add_argument("--pad_num", type=int, default=999)
    parser.add_argument("--num_test", type=int, default=2000)
    parser.add_argument("--num_train", type=int, default=8000)
    parser.add_argument("--brick_num", type=int, required=True)
    parser.add_argument("--test_modelnet3", type=int, default=0)
    parser.add_argument("--train_modelnet3", type=int, default=0)
    parser.add_argument("--object_type", type=str, required=True)
    parser.add_argument("--test_object_type", type=str, required=True)

    # network
    parser.add_argument("--view_num", type=int, default=3)
    parser.add_argument("--dec_layer", type=int, default=4)
    parser.add_argument("--enc_layer", type=int, default=4)
    parser.add_argument("--img_f_num", type=int, default=16)
    parser.add_argument("--voxel_size", type=int, default=64)
    parser.add_argument("--brick_feature_len", type=int, default=64)

    # train
    parser.add_argument("--iou_w", type=int, default=0)
    parser.add_argument("--reload", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)

    # fine tune
    parser.add_argument("--fine_tune", type=int, default=0)
    parser.add_argument("--load_model_path", type=str, default='')
    parser.add_argument("--mask_threshold", type=float, default=0.4)

    # augmentation
    parser.add_argument("--root_aug_range", type=int, default=1)
    parser.add_argument("--erase_portion", type=float, default=0)

    # iter
    parser.add_argument("--n_epochs", type=int, default=5000)
    parser.add_argument("--iou_interval", type=int, default=50)
    parser.add_argument("--ckpt_interval", type=int, default=500)
    parser.add_argument("--test_interval", type=int, default=10)

    # test
    parser.add_argument("--save_obj", type=int, default=0)
    parser.add_argument("--inference", type=int, default=0)
    parser.add_argument("--test_eval_acc", type=int, default=1)

    opt = parser.parse_args()
    opt.connect_num = def_tree.CONNECT_TYPE

    # path init
    opt.root_path = os.getcwd()
    opt.train_name = f'{opt.object_type}'
    if def_brick_type.BRICK_TYPE == '1':
        opt.train_name += '_new21'
    opt.voxels_path = os.path.join(opt.root_path, 'dataset/voxel/')
    opt.voxel_img_path = os.path.join(opt.root_path, 'dataset/voxel_img/')
    opt.act_path = os.path.join(opt.root_path, 'dataset/tree_actions/')
    opt.dep_act_path = os.path.join(opt.root_path, 'dataset/tree_dep_actions/')
    opt.model_path = os.path.join(opt.root_path, 'output/' + opt.train_name + '/model/')
    opt.test_gen_obj_path = os.path.join(opt.root_path, 'output/' + opt.train_name + '/test_gen_obj/')
    for path in [opt.model_path, opt.test_gen_obj_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    return opt
