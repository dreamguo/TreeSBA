import ipdb
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import def_tree
from utils.network_vit import ViT
from utils.def_LEGO import LEGO


class Pred_Head(nn.Module):
    def __init__(self, opt):
        super(Pred_Head, self).__init__()
        self.pred_linear = nn.Linear(opt.brick_feature_len, opt.connect_num)
        self.pred_linear_down = nn.Linear(opt.brick_feature_len, opt.connect_num)

    def forward(self, dec_output):
        actions = self.pred_linear(dec_output)
        actions_down = self.pred_linear_down(dec_output)
        return torch.stack((actions, actions_down), 1)

class My_PositionalEncoding(nn.Module):
    def test():
        pass

class TreeTransformer(nn.Module):
    def __init__(self, opt):
        super(TreeTransformer, self).__init__()
        self.device = opt.device
        self.brick_num = opt.brick_num
        self.voxel_size = opt.voxel_size
        self.dir_embed = nn.Embedding(2, opt.brick_feature_len)
        self.updown_embed = nn.Embedding(2, opt.brick_feature_len)
        self.x_embed = nn.Embedding(opt.voxel_size, opt.brick_feature_len)
        self.y_embed = nn.Embedding(opt.voxel_size, opt.brick_feature_len)
        self.z_embed = nn.Embedding(opt.voxel_size, opt.brick_feature_len)
        self.depth_embed = nn.Embedding(opt.brick_num, opt.brick_feature_len)
        self.pos_embed = nn.Embedding(def_tree.M_A2T+1, opt.brick_feature_len)
        transformer_model = nn.Transformer( batch_first=True,
                                            d_model=opt.brick_feature_len,
                                            num_encoder_layers=opt.enc_layer,
                                            num_decoder_layers=opt.dec_layer)
        self.encoder = ViT( dim=opt.brick_feature_len,
                            depth=opt.enc_layer, channels=opt.view_num)
        self.decoder = transformer_model.decoder
        self.pred_head = Pred_Head(opt)
        self.tgt_mask = self.get_query_mask(self.brick_num)

    def get_query_emb(self, pos, layer, global_pos):
        depth_emb = self.depth_embed(torch.tensor([[layer]*pos.shape[-1]], device=self.device))
        pos_emb = self.pos_embed(pos[0:1])
        updown_emb = self.updown_embed(pos[1:2])
        x_emb = self.x_embed(global_pos[0:1])
        y_emb = self.y_embed(global_pos[1:2])
        z_emb = self.z_embed(global_pos[2:3])
        dir_emb = self.dir_embed(global_pos[3:4])
        local_emb = depth_emb + pos_emb + updown_emb
        global_emb = x_emb + y_emb + z_emb + dir_emb
        return local_emb + global_emb

    def get_gt_query_emb(self, dep_w_pos):
        depth_emb = self.depth_embed(dep_w_pos[:, :, 0])
        pos_emb = self.pos_embed(dep_w_pos[:, :, 1])
        updown_emb = self.updown_embed(dep_w_pos[:, :, 2])
        x_emb = self.x_embed(dep_w_pos[:, :, 3])
        y_emb = self.y_embed(dep_w_pos[:, :, 4])
        z_emb = self.z_embed(dep_w_pos[:, :, 5])
        dir_emb = self.dir_embed(dep_w_pos[:, :, 6])
        local_emb = depth_emb + pos_emb + updown_emb
        global_emb = x_emb + y_emb + z_emb + dir_emb
        return local_emb + global_emb

    def get_query_mask(self, size):
        if size > self.brick_num:
            size = self.brick_num
        query_mask = torch.triu(torch.ones((size, size), device=self.device), diagonal=1)
        return query_mask.bool()

    def forward(self, imgs, dep_w_pos=None):
        self.device = imgs.device
        memory = self.encoder(imgs)
        if self.training:
            query_emb = self.get_gt_query_emb(dep_w_pos)
            dec_output = self.decoder(query_emb, memory, tgt_mask=self.tgt_mask)
            actions = self.pred_head(dec_output)
            return actions.argmax(dim=-1).detach(), {"res": actions.permute(0,3,1,2)}
        else:
            dec_output = self.forward_test(memory)
            actions = self.pred_head(dec_output)
            res = F.pad(actions.argmax(dim=-1), (0, self.brick_num - actions.shape[2]))
            return res

    def forward_test(self, memory):
        pos = 1
        last_pos = 0
        lego = LEGO(size=self.voxel_size, device=self.device, brick_num=self.brick_num)
        query_emb = self.get_query_emb(torch.tensor([[0],[0]],
                        device=self.device), 0, lego.coords[-1][:, None])
        for layer in range(self.brick_num):
            dec_output, nexts = self.test_pred(pos, last_pos, memory, query_emb, lego)
            if nexts.shape[-1] == 0 or pos >= self.brick_num:
                break
            last_pos = pos
            pos += nexts.shape[-1]
            tmp = []
            for i in range(last_pos, pos):
                tmp.append(lego.coords[i][:, None])
            pred_emb = self.get_query_emb(nexts, layer+1, torch.cat(tmp, dim=1))
            query_emb = torch.cat((query_emb, pred_emb), dim=1)[:, :self.brick_num, :]
        return dec_output

    def test_pred(self, pos, last_pos, memory, query_emb, lego):
        query_mask = self.get_query_mask(pos)
        dec_output = self.decoder(query_emb, memory, tgt_mask=query_mask)
        pred_f = F.log_softmax(self.pred_head(dec_output), dim=-1)[:, :, last_pos:pos, :]
        pred_actions = pred_f.argmax(dim=-1)[0]  # [0] means batch_idx == 0
        nexts = [[], []]
        for act_i in range(pred_actions.shape[1]):
            for connect_type in def_tree.MULTI_T2A[pred_actions[0][act_i].item()]:
                if lego.add_brick_fast_with_check(last_pos+act_i, connect_type, up=1):
                    nexts[0].append(connect_type)
                    nexts[1].append(0)
            for connect_type in def_tree.MULTI_T2A_DOWN[pred_actions[1][act_i].item()]:
                if lego.add_brick_fast_with_check(last_pos+act_i, connect_type, up=-1):
                    nexts[0].append(connect_type)
                    nexts[1].append(1)
        nexts = torch.tensor(nexts, device=self.device)
        return dec_output, nexts
