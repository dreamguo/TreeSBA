import os
import sys
import copy
import ipdb
import numpy as np
from tqdm import tqdm
from geometric_primitives import brick
from geometric_primitives import bricks
from geometric_primitives import utils_io as utils_gp


sys.path.append('.')
sys.path.append('./prepare_dataset')
from utils_RAD.rules import rules_2_4, rules_1_2


class My_Bricks(bricks.Bricks):
    def __init__(self, max_bricks, str_type):
        self.max_bricks = max_bricks
        self.bricks = []

        self.node_matrix = None
        self.adjacency_matrix = None
        self.edge_matrix = None
        self.degree_matrix = None

        self.str_type = str_type

        if self.str_type == '0':
            self.list_rules = rules_2_4.LIST_RULES
            self.rules = rules_2_4.RULE_CONTACTS
            self.probs_rules = rules_2_4.PROBS_CONTACTS
        elif self.str_type == '1':
            self.list_rules = rules_1_2.LIST_RULES
            self.rules = rules_1_2.RULE_CONTACTS
            self.probs_rules = rules_1_2.PROBS_CONTACTS


    def _sample_one(self, str_type=None):
        list_bricks = self.get_bricks()
        ind_brick = np.random.choice(self.get_length())
        brick_sampled = list_bricks[ind_brick]

        cur_position = brick_sampled.get_position()
        cur_direction = brick_sampled.get_direction()

        ind_rule = np.random.choice(len(self.rules), p=self.probs_rules)
        cur_rule = self.rules[ind_rule]

        translations = copy.deepcopy(cur_rule['translations'])
        direction = copy.deepcopy(cur_rule['direction'])

        ind_trans = np.random.choice(len(translations))
        trans = translations[ind_trans]

        if cur_direction == 1:
            trans[0], trans[1] = trans[1], trans[0]

        upper_lower = np.random.choice(2) * 2 - 1
        # upper_lower = 1

        if str_type == '0':
            new_brick = brick.Brick(size_upper=rules_2_4.SIZE_UPPER, 
                                    size_lower=rules_2_4.SIZE_LOWER)
        elif str_type == '1':
            new_brick = brick.Brick(size_upper=rules_1_2.SIZE_UPPER, 
                                    size_lower=rules_1_2.SIZE_LOWER)
        new_brick.set_position(cur_position + np.concatenate((np.array(trans), [new_brick.height * upper_lower])))
        new_brick.set_direction((cur_direction + direction) % 2)

        new_brick = self._validate_bricks([new_brick])

        if len(new_brick) == 0:
            return None
        else:
            return new_brick[0]


    def get_connection_type(self, brick_1, brick_2):
        # brick_1 is a yardstick.
        diff_direction = (brick_2.get_direction() + brick_1.get_direction()) % 2
        diff_position = (brick_2.get_position() - brick_1.get_position())[:2]

        if brick_1.get_direction() == 1:
            diff_position[0], diff_position[1] = diff_position[1], diff_position[0]

        ind = None
        num_ind = 0
        for rule in self.list_rules:
            if diff_direction == rule[1][0] and np.all(np.abs(diff_position - np.array(rule[1][1])) < 1e-4):
                ind = rule[0]
                num_ind += 1

        if not num_ind == 1:
            # print(brick_1.get_direction(), brick_2.get_direction())
            # print(num_ind, diff_direction, diff_position)
            raise ValueError('Invalid connection type.')

        assert ind > 0
        return ind


def create_bricks(list_bricks, str_label, dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    print('[{}] # of brick combinations: {}'.format(str_label, len(list_bricks)))

    for ind_bricks_, bricks_ in tqdm(enumerate(list_bricks)):
        utils_gp.save_bricks(bricks_, dataset_dir, str_file='{}_{:06}'.format(str_label, ind_bricks_+1))


def random(num_min, num_max, str_type):
    num_bricks = np.random.randint(low=num_min, high=num_max)
    print('Assembling {} bricks'.format(num_bricks))

    # first brick
    if str_type == '0':
        brick_ = brick.Brick(size_upper=rules_2_4.SIZE_UPPER, 
                             size_lower=rules_2_4.SIZE_LOWER)
        brick_.set_position([0, 0, 0])
        brick_.set_direction(0)
    elif str_type == '1':
        brick_ = brick.Brick(size_upper=rules_1_2.SIZE_UPPER, 
                             size_lower=rules_1_2.SIZE_LOWER)
        brick_.set_position([0, 0, 0])
        brick_.set_direction(0)

    bricks_ = My_Bricks(num_bricks, str_type)
    bricks_.add(brick_)

    for ind in range(0, num_bricks - 1):
        next_brick = None
        while next_brick is None:
            next_brick = bricks_._sample_one(str_type)
        bricks_.add(next_brick)

    # bricks_.validate_all()
    return bricks_


def make_LEGO(num_object, num_min, num_max, str_type, lego_dir, object_type):
    dataset_dir = os.path.join(lego_dir, object_type)
    os.makedirs(dataset_dir, exist_ok=True)

    for ind in tqdm(num_object):
        bricks_ = random(num_min, num_max, str_type)
        str_save = os.path.join(dataset_dir, '{}_{:06}'.format(object_type, ind+1) + '.npy')
        np.save(str_save, bricks_)


if __name__ == '__main__':
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--id', type=int)
    args = parser.parse_args()
    
    str_type = '1'  # '0': 2x4 brick, '1': 1x2 brick
    num_min = 60
    num_max = 200
    num_object_l = []
    for i in range(60):
        num_object_l.append(range(500*i, 500*(i+1)))
    num_object = num_object_l[args.id]
    print(num_object)
    object_type = 'random{}-{}to{}'.format(str_type, num_min, num_max)
    ROOT_PATH = os.path.join(os.getcwd(), 'dataset')
    lego_dir = os.path.join(ROOT_PATH, 'LEGO')

    make_LEGO(num_object, num_min, num_max, str_type, lego_dir, object_type)
