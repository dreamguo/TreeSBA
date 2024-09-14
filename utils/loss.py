#!/usr/bin/env python
# coding=utf-8
import ipdb
import torch
from torch.nn import functional as F

from utils import def_space


def criterion(feature_dict, gt_acts, gt_dep_acts, gt_imgs, opt, gt_voxels):
    if not opt.fine_tune:
        CE_loss = torch.nn.CrossEntropyLoss(ignore_index=opt.pad_num)
        act_loss = CE_loss(feature_dict['res'], gt_acts)
        occupy_loss = torch.tensor(0)
    else:
        act_loss = torch.tensor(0)
        pred_v = get_pred_voxel_prob(feature_dict, gt_dep_acts, opt)
        occupy_loss = get_img_loss(pred_v, gt_imgs, opt.iou_w)
    return act_loss, occupy_loss


def get_pred_voxel_prob(feature_dict, gt_dep_acts, opt):
    updown_n = 2
    batch_n, act_n, _ = gt_dep_acts.shape
    gt_pos = gt_dep_acts[:, :, 3:6]
    gt_dir = gt_dep_acts[:, :, 6]
    gt_pos_ = gt_pos.repeat(1,1,updown_n).reshape(batch_n,act_n,updown_n,1,3)
    src_mask = (gt_pos>=2).any(dim=2) & (gt_pos<opt.voxel_size-2).all(dim=2)
    src_mask = src_mask.repeat(1,updown_n).reshape(batch_n,updown_n,act_n)
    pred_v = torch.zeros((batch_n, act_n, updown_n, opt.voxel_size,
                            opt.voxel_size, opt.voxel_size), device=opt.device)
    feature_prob = F.softmax(feature_dict['res'], dim=1)

    stud_n = 4 * def_space.STUD_N
    connect_i = 0
    b_grid, a_grid, u_grid, _ = torch.meshgrid(    
        torch.arange(batch_n, device=opt.device),
        torch.arange(act_n, device=opt.device),
        torch.arange(updown_n, device=opt.device),
        torch.arange(stud_n, device=opt.device), indexing='ij')
    index_ = gt_pos_.repeat(1,1,1,stud_n,1) + \
            def_space.MULTI_OFFSET[gt_dir][:,:,:,connect_i,:stud_n,:].to(opt.device)
    index_ = index_.clamp(min=0, max=opt.voxel_size-1)
    index = (   b_grid.flatten(),
                a_grid.flatten(),
                u_grid.flatten(),
                index_[..., 0].flatten(),
                index_[..., 1].flatten(),
                index_[..., 2].flatten()    )
    src = feature_prob[:, connect_i] * src_mask
    src = src.detach() - src
    src = src.transpose(1,2)[..., None].repeat(1,1,1,stud_n)
    pred_v = pred_v.index_put(index, src.flatten(), accumulate=True)

    stud_n = def_space.STUD_N
    b_grid, a_grid, u_grid, _ = torch.meshgrid(    
        torch.arange(batch_n, device=opt.device),
        torch.arange(act_n, device=opt.device),
        torch.arange(updown_n, device=opt.device),
        torch.arange(stud_n, device=opt.device), indexing='ij')
    for connect_i in range(1, def_space.MAX_T):
        index_ = gt_pos_.repeat(1,1,1,stud_n,1) + \
                def_space.MULTI_OFFSET[gt_dir][:,:,:,connect_i,:stud_n,:].to(opt.device)
        index_ = index_.clamp(min=0, max=opt.voxel_size-1)
        index = (   b_grid.flatten(),
                    a_grid.flatten(),
                    u_grid.flatten(),
                    index_[..., 0].flatten(),
                    index_[..., 1].flatten(),
                    index_[..., 2].flatten()    )
        src = feature_prob[:, connect_i] * src_mask
        src = src.transpose(1,2)[..., None].repeat(1,1,1,stud_n)
        pred_v = pred_v.index_put(index, src.flatten(), accumulate=True)

    stud_n = 2 * def_space.STUD_N
    b_grid, a_grid, u_grid, _ = torch.meshgrid(    
        torch.arange(batch_n, device=opt.device),
        torch.arange(act_n, device=opt.device),
        torch.arange(updown_n, device=opt.device),
        torch.arange(stud_n, device=opt.device), indexing='ij')
    for connect_i in range(def_space.MAX_T, opt.connect_num):
        index_ = gt_pos_.repeat(1,1,1,stud_n,1) + \
                def_space.MULTI_OFFSET[gt_dir][:,:,:,connect_i,:stud_n,:].to(opt.device)
        index_ = index_.clamp(min=0, max=opt.voxel_size-1)
        index = (   b_grid.flatten(),
                    a_grid.flatten(),
                    u_grid.flatten(),
                    index_[..., 0].flatten(),
                    index_[..., 1].flatten(),
                    index_[..., 2].flatten()    )
        src = feature_prob[:, connect_i] * src_mask
        src = src.transpose(1,2)[..., None].repeat(1,1,1,stud_n)
        pred_v = pred_v.index_put(index, src.flatten(), accumulate=True)

    pred_v[pred_v < opt.mask_threshold] = 0
    pred_v = pred_v.sum(dim=[1,2]) / (pred_v!=0).sum(dim=[1,2]).clamp(1)
    return pred_v


def get_img_loss(pred_v, gt_imgs, iou_w):
    pred_imgs = torch.ones_like(gt_imgs)
    for i in range(gt_imgs.shape[1]):
        pred_imgs[:, i] = pred_v.sum(dim=[i+1]) / (pred_v!=0).sum(dim=[i+1]).clamp(1)
    pred_imgs[pred_imgs>1] -= (pred_imgs[pred_imgs>1].detach() - 1)

    BCE_loss = torch.nn.BCELoss(reduction='none')
    bce_loss = BCE_loss(pred_imgs, gt_imgs).sum(dim=[1,2,3])
    bce_loss /= torch.count_nonzero(gt_imgs, dim=[1,2,3])
    bce_loss = bce_loss.mean()

    DICE_loss = DiceLoss()
    iou_loss = DICE_loss(pred_imgs, gt_imgs)
    return bce_loss + iou_w * iou_loss


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
 
    def	forward(self, pred, target):
        smooth = 1
        N = target.size(0)
        pred_flat = pred.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = pred_flat * target_flat
        loss = (2 * intersection.sum(1) + smooth) / (
            pred_flat.sum(1) + target_flat.sum(1) + smooth)
        return 1 - loss.mean()
