
import os
import ipdb
import torch
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import open3d as o3d
import polyscope as ps

from utils import def_brick_type, def_tree
from utils.def_LEGO import LEGO


PATH_LEGO_2_4 = 'unit_primitives/lego_2_4.stl'
PATH_LEGO_1_2 = 'unit_primitives/lego_1_2.stl'
PATH_LEGO_1_1 = 'unit_primitives/lego_1_1.stl'
if def_brick_type.BRICK_TYPE == '0':
    PATH_LEGO = PATH_LEGO_2_4
    VERTS_N = 2564
    FACE_N = 2018
elif def_brick_type.BRICK_TYPE == '1':
    PATH_LEGO = PATH_LEGO_1_2
    VERTS_N = 821
DEBUG = 0


def get_tree_voxel(actions, voxel_size, pad_num, brick_num):
    lego = LEGO(size=voxel_size, device=actions.device, brick_num=brick_num)
    node_queue = [0]
    neighbor_node_idx = 1
    while len(node_queue) > 0:
        node_idx = node_queue.pop(0)
        if node_idx >= actions.shape[1] or actions[0][node_idx] == pad_num:
            break
        connect_types = def_tree.MULTI_T2A[actions[0][node_idx].item()]
        for connect_type in connect_types:
            if lego.add_brick_fast_with_check(node_idx, connect_type, up=1):
                node_queue.append(neighbor_node_idx)
                neighbor_node_idx += 1
        connect_types = def_tree.MULTI_T2A_DOWN[actions[1][node_idx].item()]
        for connect_type in connect_types:
            if lego.add_brick_fast_with_check(node_idx, connect_type, up=-1):
                node_queue.append(neighbor_node_idx)
                neighbor_node_idx += 1
    return lego.voxel


def get_brick_num(actions, opt):
    lego = LEGO(size=opt.voxel_size, device=actions.device, brick_num=opt.brick_num)
    node_queue = [0]
    neighbor_node_idx = 1
    while len(node_queue) > 0:
        node_idx = node_queue.pop(0)
        if node_idx >= actions.shape[1] or actions[0][node_idx] == opt.pad_num:
            break
        connect_types = def_tree.MULTI_T2A[actions[0][node_idx].item()]
        for connect_type in connect_types:
            if lego.add_brick_fast_with_check(node_idx, connect_type, up=1):
                node_queue.append(neighbor_node_idx)
                neighbor_node_idx += 1
        connect_types = def_tree.MULTI_T2A_DOWN[actions[1][node_idx].item()]
        for connect_type in connect_types:
            if lego.add_brick_fast_with_check(node_idx, connect_type, up=-1):
                node_queue.append(neighbor_node_idx)
                neighbor_node_idx += 1
    return len(lego.coords)


def get_tree_voxels(actions, opt):
    voxels = []
    for act in actions:
        voxel = get_tree_voxel(act, opt.voxel_size, opt.pad_num, opt.brick_num)
        voxels.append(voxel[None])
    voxels = torch.cat(voxels)
    return voxels


def eval_img_iou(gt_voxels, pred_voxels):
    gt_imgs = []
    pred_imgs = []
    for view_i in range(1, 4):
        gt_imgs.append(gt_voxels.any(axis=view_i))
        pred_imgs.append(pred_voxels.any(axis=view_i))
    gt_imgs = torch.stack(gt_imgs, dim=1)
    pred_imgs = torch.stack(pred_imgs, dim=1)
    pred_one = pred_imgs.eq(1)
    match = torch.sum(torch.logical_and(pred_imgs.eq(gt_imgs), pred_one), axis=[2,3])
    iou_img = match / (torch.sum(pred_one, axis=[2,3]) +
                torch.sum(gt_imgs.eq(1), axis=[2,3]) - match)
    return iou_img


def eval_3d_iou(pred_act, gt_act, gt_voxels, opt):
    pred_voxels = get_tree_voxels(pred_act.cpu(), opt)
    if gt_voxels.shape != pred_voxels.shape:
        gt_voxels = get_tree_voxels(gt_act.cpu(), opt)
    pred_one = pred_voxels.eq(1)
    match = torch.sum(torch.logical_and(pred_voxels.eq(gt_voxels), pred_one), axis=[1,2,3])
    iou_3d = match / (torch.sum(pred_one, axis=[1,2,3]) +
            torch.sum(gt_voxels.eq(1), axis=[1,2,3]) - match)
    return iou_3d, eval_img_iou(gt_voxels, pred_voxels)


def eval_act(pred_act, gt_act, opt):
    matches = pred_act.eq(gt_act).sum(dim=[1, 2])
    valid_len = opt.brick_num * 2 - gt_act.eq(opt.pad_num).sum(dim=[1, 2])
    return matches / valid_len


def eval_func(pred_act, gt_act, gt_voxels, opt, eval_acc, eval_iou):
    if eval_acc:
        tmp_gt = gt_act + 0
        tmp_gt[gt_act.eq(opt.pad_num)] = 0
        success = (tmp_gt == pred_act).all().int().item()
    else:
        success = 0
    acc_act = eval_act(pred_act, gt_act, opt) if eval_acc else torch.tensor([0.0])
    if eval_iou:
        iou_3d, iou_img = eval_3d_iou(pred_act, gt_act, gt_voxels, opt)
    else:
        iou_3d, iou_img = torch.tensor([0.0]), torch.tensor([0.0])
    return acc_act, iou_3d, iou_img, success
 

def preprocess(model):
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = (max_bound + min_bound) / 2.0
    scale = np.linalg.norm(max_bound - min_bound) / 2.0
    vertices = np.asarray(model.vertices)
    vertices -= np.matlib.repmat(center, len(model.vertices), 1)
    model.vertices = o3d.utility.Vector3dVector(vertices / scale)
    return model


def get_mesh_from_tree_actions(seqs, opt):
    brick_num = opt.brick_num
    mesh_brick = o3d.io.read_triangle_mesh(os.path.join(opt.root_path, PATH_LEGO))
    if def_brick_type.BRICK_TYPE == '0':
        scale = 25
    if def_brick_type.BRICK_TYPE == '1':
        R = mesh_brick.get_rotation_matrix_from_xyz((0, np.pi / 2, -np.pi / 2))
        mesh_brick.rotate(R)
        if DEBUG:
            ipdb.set_trace()
            o3d.visualization.draw_geometries([mesh_brick])
        scale = 100
    bound_min = mesh_brick.get_min_bound()
    bound_max = mesh_brick.get_max_bound()

    center = (bound_max + bound_min) / 2
    vertices = np.array(mesh_brick.vertices)
    vertices -= center
    vertices /= scale
    width = vertices[..., 0].max() - vertices[..., 0].min()
    length = vertices[..., 1].max() - vertices[..., 1].min()
    if def_brick_type.BRICK_TYPE == '0':
        offset = width / 2
    elif def_brick_type.BRICK_TYPE == '1':
        offset = width
    height = vertices[..., 2].max() - vertices[..., 2].min()
    # 4 / 28 is the height of stud
    height *= 6/7

    triangles = np.array(mesh_brick.triangles)
    R = [[0, 1, 0],
         [-1, 0, 0],
         [0, 0, 1]]
    R = torch.tensor(R, device=opt.device, dtype=torch.float32)

    node_queue = [0]
    root_type = 0
    neighbor_node_idx = 1
    t_all_pos = torch.zeros(brick_num, 4, device=opt.device)

    # first brick with first neighbor
    t_pos = torch.tensor([0,0,0], device=opt.device, dtype=torch.float32)
    t_vertices = torch.tensor(vertices.tolist(), device=opt.device, dtype=torch.float32)
    t_triangles = torch.tensor(triangles.tolist(), device=opt.device, dtype=torch.int32)
    if root_type == 1:
        t_vertices = torch.mm(t_vertices, R)
    t_all_pos[0][:3] = t_pos
    t_all_pos[0][3] = root_type
    all_vertices = [t_vertices]
    all_triangles = [t_triangles]

    lego = LEGO(size=opt.voxel_size, device=opt.device, brick_num=brick_num)
    if DEBUG:
        print(seqs)
    while len(node_queue) > 0:
        node_idx = node_queue.pop(0)
        if node_idx >= seqs.shape[1]:
            break

        pos_l = []
        t_direc_l = []
        connect_types = def_tree.MULTI_T2A[seqs[0][node_idx].item()]
        if DEBUG:
            print('up ', connect_types)
        for connect_type in connect_types:
            if lego.add_brick_fast_with_check(node_idx, connect_type, up=1):
                t_direc, pos = def_tree.TREE2ACTION[connect_type]
                pos_l.append(pos)
                t_direc_l.append(t_direc)
        connect_types = def_tree.MULTI_T2A_DOWN[seqs[1][node_idx].item()]
        if DEBUG:
            print('down ', connect_types)
        for connect_type in connect_types:
            if lego.add_brick_fast_with_check(node_idx, connect_type, up=-1):
                t_direc, pos = def_tree.TREE2ACTION[connect_type]
                pos = list(pos)
                pos[2] = -1
                pos_l.append(pos)
                t_direc_l.append(t_direc)

        for t_direc, pos in zip(t_direc_l, pos_l):
            if neighbor_node_idx >= brick_num:
                break
            t_pos = torch.tensor(pos, device=opt.device, dtype=torch.float32)
            t_pos[:2] *= offset

            # add edge to previous one on which position and oritation
            triangles += vertices.shape[0]
            t_vertices = torch.tensor(vertices.tolist(), device=opt.device, dtype=torch.float32)
            t_triangles = torch.tensor(triangles.tolist(), device=opt.device, dtype=torch.int32)

            # local framework to global framwwork
            if t_all_pos[node_idx][3] == 1:
                pos_x = t_pos[0].clone()
                t_pos[0] = t_pos[1]
                t_pos[1] = -pos_x
            t_pos[:2] += t_all_pos[node_idx][:2]
            assert t_pos[2] == 1 or t_pos[2] == -1
            t_pos[2] = t_all_pos[node_idx][2] + t_pos[2] * height

            t_direc = t_direc ^ int(t_all_pos[node_idx][3])
            if t_direc == 1:
                t_vertices = torch.mm(t_vertices, R)
            t_vertices += t_pos

            t_all_pos[neighbor_node_idx][:3] = t_pos
            t_all_pos[neighbor_node_idx][3] = t_direc
            all_vertices.append(t_vertices)
            all_triangles.append(t_triangles)
            node_queue.append(neighbor_node_idx)
            neighbor_node_idx += 1
        if DEBUG:
            print('node_queue ', node_queue)
            ipdb.set_trace()

    return all_vertices, all_triangles


def get_mesh_from_voxel(voxel, opt):
    mesh_brick = o3d.io.read_triangle_mesh(os.path.join(opt.root_path, PATH_LEGO_1_1))
    bound_min = mesh_brick.get_min_bound()
    bound_max = mesh_brick.get_max_bound()
    scale = 100

    vertices = np.array(mesh_brick.vertices)
    vertices /= scale
    width = vertices[..., 0].max() - vertices[..., 0].min()
    height = vertices[..., 1].max() - vertices[..., 1].min()
    # 4 / 28 is the height of stud
    height *= 6/7

    triangles = np.array(mesh_brick.triangles)
    R = [[1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]]
    R = torch.tensor(R, device=opt.device, dtype=torch.float32)

    t_vertices = torch.tensor(vertices.tolist(), device=opt.device, dtype=torch.float32)
    t_triangles = torch.tensor(triangles.tolist(), device=opt.device, dtype=torch.int32)
    t_vertices = torch.mm(t_vertices, R)
    all_vertices = t_vertices
    all_triangles = t_triangles

    x, y, z = torch.where(voxel)
    axis = torch.concat((x[None] - 32, y[None] - 32, z[None]- 32)).permute(1, 0).float()
    for axi in axis:
        triangles += vertices.shape[0]
        t_vertices = torch.tensor(vertices.tolist(), device=opt.device, dtype=torch.float32)
        t_triangles = torch.tensor(triangles.tolist(), device=opt.device, dtype=torch.int32)
        t_vertices = torch.mm(t_vertices, R)
        t_vertices += axi * width
        all_vertices = torch.cat((all_vertices, t_vertices), 0)
        all_triangles = torch.cat((all_triangles, t_triangles), 0)
    return all_vertices, all_triangles

def get_mesh_bricks(bricks_, device, root_path):
    mesh_brick = o3d.io.read_triangle_mesh(os.path.join(root_path, PATH_LEGO))
    mesh_brick = preprocess(mesh_brick)
    
    bound_min = mesh_brick.get_min_bound()
    bound_max = mesh_brick.get_max_bound()
    unit_axis_1 = (bound_max[0] - bound_min[0]) / 2
    unit_axis_2 = (bound_max[1] - bound_min[1]) / 4
    unit_axis_3 = (bound_max[2] - bound_min[2])
    # 4 / 28 is the height of stud
    unit_axis_3 *= 6/7
    R = [[0, 1, 0],
         [-1, 0, 0],
         [0, 0, 1]]
    R = torch.tensor(R, device=device, dtype=torch.float32)
    vertices = np.array(mesh_brick.vertices)
    triangles = np.array(mesh_brick.triangles)
    triangles -= vertices.shape[0]
    
    all_vertices = []
    all_triangles = []
    for brick_idx, brick in enumerate(bricks_.get_bricks()):
        triangles += vertices.shape[0]
        t_triangles = torch.tensor(triangles.tolist(), device=device, dtype=torch.int32)
        t_vertices = torch.tensor(vertices.tolist(), device=device, dtype=torch.float32)

        pos = brick.get_position()
        direc = brick.get_direction()
        translation = np.array([unit_axis_1 * pos[0],
                                unit_axis_2 * pos[1],
                                unit_axis_3 * pos[2]    ])
        t_vertices += torch.tensor(translation, device=device, dtype=torch.float32)
        if direc == 1:
            center = torch.mean(t_vertices, axis=0)
            t_vertices -= center
            t_vertices = torch.mm(t_vertices, R)
            t_vertices += center

        all_vertices.append(t_vertices)
        all_triangles.append(t_triangles)
    all_vertices = torch.cat(all_vertices, 0)
    all_triangles = torch.cat(all_triangles, 0)
    
    return all_vertices, all_triangles


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
