import os
import sys
import ast
import ipdb
import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from utils import def_tree
from utils.def_LEGO import LEGO
from GenerativeLEGO.pyFiles import LDraw


class LegoDataset(Dataset):
    def __init__(self, config = None):
        super().__init__()
        self.root_path = os.getcwd() + '/dataset/'
        self.dataset = []
        try:
            dataset_with_filenames = self.load_dataset_from_file(config)
            print('loaded dataset from file')
        except FileNotFoundError:
            print('making dataset')
            dataset_with_filenames = self.make_dataset_from_LDraw(config)
            self.write_dataset_to_file(config, dataset_with_filenames)
        self.add_dataset(dataset_with_filenames)

    def load_dataset_from_file(self, config):
        pickle_file = self.root_path + 'graph_dat/{}.dat'.format(config['object_type'])
        with open(pickle_file, 'rb') as f:
            dataset_with_filenames = pickle.load(f)
        return dataset_with_filenames

    def make_dataset_from_LDraw(self, config):
        dataset_with_filenames = []
        directory = os.fsencode(os.path.join(self.root_path, 'ldr', config['object_type']))
        for file in sorted(os.listdir(directory)):
            filename = os.fsdecode(file)
            print("ldr2graph", filename)
            directory_str = os.fsdecode(directory) + '/'
            try:
                lg = LDraw.LDraw_to_graph(directory_str + filename)
                dataset_with_filenames.append([filename, lg])
            except:
                continue
        return dataset_with_filenames

    def write_dataset_to_file(self, config, dataset_with_filenames):
        try:
            os.mkdir(os.path.join(os.getcwd(), 'dataset'))
        except FileExistsError:
            pass

        pickle_file = self.root_path + 'graph_dat/{}.dat'.format(config['object_type'])
        with open(pickle_file, 'wb') as f:
            pickle.dump(dataset_with_filenames, f)

    def add_dataset(self, dataset_with_filenames):
        for graph_with_filename in tqdm(dataset_with_filenames):
            filename = graph_with_filename[0]
            graph = graph_with_filename[1]
            self.add_graph_to_dataset(graph, filename.split('.')[0])

    def __len__(self):
        return len(self.dataset)

    def collate_single(self, batch):
        assert len(batch) == 1, 'Currently we do not support batched training'
        return batch[0]

    def collate_batch(self, batch):
        act = []
        imgs = []
        for item in batch:
            act.append(item[0])
            imgs.append(item[1][None])
        return act, torch.cat(imgs)

def save_action(graph, brick_num, pad_num, voxel_size,
                        save_dir, dep_save_dir, name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(dep_save_dir):
        os.makedirs(dep_save_dir)
    for root in range(brick_num):
        if graph.node_labels.get(root) == None:
            continue
        root_dir = int(graph.node_labels.get(root) == 'Brick(4, 2)')
        node_queue = [root]
        found_node_set = {root}
        act = [[], []]
        while len(node_queue) > 0:
            node = node_queue.pop(0)
            nodes, found_node_set, act[0] = get_action(
                graph, node, found_node_set, act[0], root_dir)
            node_queue.extend(nodes)
            nodes, found_node_set, act[1] = get_action(
                graph, node, found_node_set, act[1], root_dir, False)
            node_queue.extend(nodes)

        # assert graph.node_labels[0] == 'Brick(2, 4)' # x = [0,2], y = [0,4]
        size = ((0,0), (0, brick_num - len(act[0])))
        act = np.pad(act, size, 'constant', constant_values=pad_num)
        npy_name = name + '_root' + str(root) + '.npy'
        dep_act = get_dep_action(act, pad_num, voxel_size, brick_num)
        np.save(os.path.join(save_dir, npy_name), act)
        np.save(os.path.join(dep_save_dir, npy_name), dep_act)

def get_dep_action(act, pad_num, voxel_size, brick_num):
    idx = 1
    lego = LEGO(size=voxel_size, device='cpu', brick_num=brick_num)
    # [depth, connect_type, updown, x, y, z, dir]
    dep_act = np.zeros((act.shape[1], 7), np.int64)
    dep_act[0, 3:] = lego.coords[-1].tolist()
    for prev_i in range(0, act.shape[1]):
        if act[0][prev_i] == pad_num:
            break
        for connect_type in def_tree.MULTI_T2A[act[0][prev_i]]:
            if lego.add_brick_fast_with_check(prev_i, connect_type, up=1):
                dep_act[idx][0] = dep_act[prev_i][0] + 1
                dep_act[idx][1] = connect_type
                dep_act[idx][2] = 0
                dep_act[idx][3:] = lego.coords[-1].tolist()
                idx += 1
        for connect_type in def_tree.MULTI_T2A_DOWN[act[1][prev_i]]:
            if lego.add_brick_fast_with_check(prev_i, connect_type, up=-1):
                dep_act[idx][0] = dep_act[prev_i][0] + 1
                dep_act[idx][1] = connect_type
                dep_act[idx][2] = 1
                dep_act[idx][3:] = lego.coords[-1].tolist()
                idx += 1
    return dep_act

def clockwise_90(edge_embedding):
    return (-edge_embedding[1], edge_embedding[0])

def get_action(graph, node, found_node_set, act, root_dir, growup=True):
    connect_types = []
    neighbors = np.array([], dtype=np.int64)
    if growup:
        for nb in graph.g_directed.neighbors(node):
            neighbors = np.append(neighbors, nb)
    else:
        for nb in graph.g_undirected.neighbors(node):
            if nb not in graph.g_directed.neighbors(node):
                neighbors = np.append(neighbors, nb)
    for neighbor in neighbors:
        rotation = graph.node_labels[node] == graph.node_labels[neighbor]
        if growup:
            edge_embedding = ast.literal_eval(graph.edge_labels[node, neighbor])
        else:
            edge_embedding = ast.literal_eval(graph.edge_labels[neighbor, node])
            edge_embedding = (-edge_embedding[0], -edge_embedding[1])
        if root_dir == 1:
            edge_embedding = clockwise_90(edge_embedding)
        if root_dir == 0 and graph.node_labels[node] == 'Brick(4, 2)' or \
            root_dir == 1 and graph.node_labels[node] == 'Brick(2, 4)':
            edge_embedding = clockwise_90(edge_embedding)
        connect_types.append(def_tree.ACTION2TREE[rotation][edge_embedding])
        # try:
        #     connect_types.append(tree_def.ACTION2TREE[rotation][edge_embedding])
        # except:
        #     neighbors = np.setdiff1d(neighbors, neighbor)
    sort_idx = np.argsort(connect_types)
    connect_types = np.sort(connect_types)
    nodes = []
    for neighbor in neighbors[sort_idx]:
        if neighbor not in found_node_set:
            nodes.append(neighbor)
    found_node_set.update(neighbors)

    if growup:
        act.append(def_tree.MULTI_A2T[tuple(connect_types)])
    else:
        act.append(def_tree.MULTI_A2T_DOWN[tuple(connect_types)])
    return nodes, found_node_set, act


class GetDGMGDataset(LegoDataset):
    def __init__(self, config=None):
        self.config = config
        self.img_dataset = []
        super().__init__(config = config)

    def add_graph_to_dataset(self, graph, filename):
        save_dir = os.path.join(self.root_path, 'tree_actions', self.config['object_type'])
        dep_save_dir = os.path.join(self.root_path, 'tree_dep_actions', self.config['object_type'])
        save_action(graph, self.config['brick_num'], self.config['pad_num'],
                    self.config['voxel_size'], save_dir, dep_save_dir, filename)

    def __getitem__(self, index):
        return self.dataset[index], self.img_dataset[index]
