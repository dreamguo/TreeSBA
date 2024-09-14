#!/usr/bin/env python
# coding=utf-8

import ipdb
import torch

from utils import eval_utils


def seq2vt(seqs_list, opt, return_list=0):
    # change view
    R_x = [[1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]]
    R_x = torch.tensor(R_x, device=opt.device, dtype=torch.float32)

    t_verts_list = []
    t_triags_list = []
    for idx, seqs in enumerate(seqs_list):
        t_vertices, t_triangles = eval_utils.get_mesh_from_tree_actions(seqs, opt)
        if return_list:
            for i in range(len(t_vertices)):
                t_vertices[i] = torch.mm(t_vertices[i], R_x)
        else:
            t_vertices = torch.cat(t_vertices)
            t_triangles = torch.cat(t_triangles)
            t_vertices = torch.mm(t_vertices, R_x)

        t_verts_list.append(t_vertices)
        t_triags_list.append(t_triangles)
    return t_verts_list, t_triags_list

def voxel2vt(voxel_list, opt):
    t_verts_list = []
    t_triags_list = []
    for idx, voxel in enumerate(voxel_list):
        t_vertices, t_triangles = eval_utils.get_mesh_from_voxel(voxel, opt)
        # change view
        R_x = [[1, 0, 0],
               [0, 0, -1],
               [0, 1, 0]]
        R_x = torch.tensor(R_x, device=voxel.device, dtype=torch.float32)
        t_vertices = torch.mm(t_vertices, R_x)

        t_verts_list.append(t_vertices)
        t_triags_list.append(t_triangles)
    return t_verts_list, t_triags_list
