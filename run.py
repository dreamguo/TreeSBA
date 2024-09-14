import os
import ipdb
import wandb
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from utils import def_brick_type
def_brick_type.BRICK_TYPE = '0'

from config import config_parser
from utils import network, datasets, eval_utils, loss, render, helpers

cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'


def main():
    # init model & dataloader
    if not opt.inference:
        traindataset = datasets.TrainSet(opt=opt)
        traindataloader = DataLoader(traindataset,
                                    batch_size=opt.batch_size,
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=opt.num_workers)
    testdataset = datasets.TestSet(opt=opt)
    testdataloader = DataLoader(testdataset, num_workers=opt.num_workers)
    model = network.TreeTransformer(opt).cuda() if cuda else network.TreeTransformer(opt)

    start_epoch = 0
    # load model
    if opt.load_model_path != '':
        print('load model from ', opt.load_model_path)
        load_info = torch.load(opt.load_model_path, map_location=opt.device)
        model = load_info['model']
        if opt.reload:
            start_epoch = load_info['epoch']

    # inference
    if opt.inference:
        print('testset {}'.format(len(testdataset)))
        test_epoch(testdataloader, model, 0)
        return

    # train
    print('trainset {}, testset {}'.format(len(traindataset), len(testdataset)))
    if opt.use_wandb:
        wandb.init(project="TreeSBA", entity="entity_name", config=opt, name=opt.train_name)

    # core loop
    print('-' * 40, ' start ', '-' * 40)
    optimizer = AdamW(model.parameters(), lr=opt.lr)
    for epoch in range(start_epoch, opt.n_epochs):
        if epoch % opt.test_interval == 0:
            dict_wandb = test_epoch(testdataloader, model, epoch)
        if (epoch != 0 and epoch % opt.ckpt_interval == 0):
            ckpt = {"model": model,
                    "epoch": epoch,
                    "dict_wandb": dict_wandb}
            torch.save(ckpt, os.path.join(opt.model_path, 'model' + str(epoch) + '.pt'))
        if opt.fine_tune:
            dict_wandb = fine_tune_epoch(traindataloader, model, optimizer, epoch, dict_wandb)
        else:
            dict_wandb = train_epoch(traindataloader, model, optimizer, epoch, dict_wandb)
        if opt.use_wandb:
            wandb.log(dict_wandb)


def train_epoch(traindataloader, model, optimizer, epoch, dict_wandb):
    model.train()
    train_accs = []
    train_iou_3d = []
    train_iou_img = []
    act_loss_epoch = 0
    for (gt_imgs, gt_act, gt_dep_act, gt_voxels, names) in tqdm(traindataloader):
        gt_act = gt_act.to(opt.device)
        gt_imgs = gt_imgs.to(opt.device)
        gt_dep_act = gt_dep_act.to(opt.device)

        # Generate a batch of actions
        optimizer.zero_grad()
        pred_act, feature_dict = model(gt_imgs, gt_dep_act)
        act_loss, occupy_loss = loss.criterion(feature_dict, gt_act, 
                                                gt_dep_act, gt_imgs, opt, gt_voxels)
        act_loss_epoch += act_loss.item()
        act_loss.backward()
        optimizer.step()

        # eval metrics
        accs, iou_3d, iou_img, _ = eval_utils.eval_func(pred_act, gt_act, gt_voxels,
            opt, eval_acc=True, eval_iou=(epoch != 0 and epoch % opt.iou_interval == 0))
        train_accs.append(accs)
        train_iou_3d.append(iou_3d)
        train_iou_img.append(iou_img)

    act_loss_epoch /= len(traindataloader)
    acc_mean = torch.mean(torch.cat(train_accs))
    iou_3d_mean = torch.mean(torch.cat(train_iou_3d))
    iou_img_mean = torch.mean(torch.cat(train_iou_img))
    print(  "Epoch %d [act_loss: %f] [iou_3d: %f] [iou_img: %f] [acc: %f]" %
            (epoch, act_loss_epoch, iou_3d_mean, iou_img_mean, acc_mean)  )
    dict_wandb.update({ "act_loss": act_loss_epoch,
                        "acc": acc_mean,    })
    if iou_3d_mean != 0:
        dict_wandb["iou"] = iou_3d_mean
        dict_wandb["iou_img"] = iou_img_mean
    return dict_wandb


def fine_tune_epoch(traindataloader, model, optimizer, epoch, dict_wandb):
    train_iou_3d = []
    train_iou_img = []
    occupy_loss_epoch = 0
    for (gt_imgs, gt_act, gt_dep_act, gt_voxels, names) in tqdm(traindataloader):
        gt_imgs = gt_imgs.to(opt.device)

        with torch.no_grad():
            model.eval()
            pred_dep_act = []
            for gt_img in gt_imgs:
                pred_act, *_ = model(gt_img[None])
                pred_dep_act_np = helpers.get_dep_action(pred_act.cpu().numpy(),
                                    opt.pad_num, opt.voxel_size, opt.brick_num)
                pred_dep_act.append(torch.tensor(pred_dep_act_np).to(opt.device)[None])
            pred_dep_act = torch.cat(pred_dep_act)

        # Generate a batch of actions
        model.train()
        optimizer.zero_grad()
        pred_act, feature_dict = model(gt_imgs, pred_dep_act)

         # eval metrics
        accs, iou_3d, iou_img, _ = eval_utils.eval_func(pred_act, gt_act, gt_voxels,
            opt, eval_acc=False, eval_iou=(epoch % opt.iou_interval == 0))
        train_iou_3d.append(iou_3d)
        train_iou_img.append(iou_img)

        act_loss, occupy_loss = loss.criterion(feature_dict, gt_act, 
                                                pred_dep_act, gt_imgs, opt, gt_voxels)
        occupy_loss_epoch += occupy_loss.item()
        occupy_loss.backward()
        optimizer.step()

    occupy_loss_epoch /= len(traindataloader)
    iou_3d_mean = torch.mean(torch.cat(train_iou_3d))
    iou_img_mean = torch.mean(torch.cat(train_iou_img))
    print(  "Epoch %d [occupy_loss: %f] [iou_3d: %f] [iou_img: %f]" %
            (epoch, occupy_loss_epoch, iou_3d_mean, iou_img_mean)  )
    dict_wandb.update({ "occupy_loss": occupy_loss_epoch})
    if iou_3d_mean != 0:
        dict_wandb["iou"] = iou_3d_mean
        dict_wandb["iou_img"] = iou_img_mean
    return dict_wandb


@torch.no_grad()
def test_epoch(testdataloader, model, epoch):
    model.eval()
    successes = []
    test_accs = []
    test_iou_3d = []
    test_iou_img = []
    if opt.save_obj:
        print('save in the', opt.test_gen_obj_path)
    for batch_idx, (gt_imgs, gt_act, gt_dep_act, gt_voxels, names) in enumerate(tqdm(testdataloader)):
        gt_act = gt_act.to(opt.device)
        gt_imgs = gt_imgs.to(opt.device)
        gt_dep_act = gt_dep_act.to(opt.device)

        # Generate a batch of actions
        pred_act, *_ = model(gt_imgs, gt_dep_act)

        # eval metrics
        t_accs, t_iou_3d, t_iou_img, success = eval_utils.eval_func(pred_act[None], gt_act, gt_voxels, 
            opt, eval_acc=opt.test_eval_acc, eval_iou=(epoch % opt.iou_interval == 0))
        successes.append(success)
        test_accs.append(t_accs)
        test_iou_3d.append(t_iou_3d)
        test_iou_img.append(t_iou_img)

    success_rate = np.mean(successes)
    acc_mean = torch.mean(torch.cat(test_accs))
    iou_3d_mean = torch.mean(torch.cat(test_iou_3d))
    iou_img_mean = torch.mean(torch.cat(test_iou_img))
    print(  "[t_iou_3d: %f] [t_iou_img: %f] [t_acc: %f] [suc_rate: %f]" %
            (iou_3d_mean, iou_img_mean, acc_mean, success_rate)  )
    dict_wandb = { "test_acc": acc_mean}
    if iou_3d_mean != 0:
        dict_wandb["test_iou"] = iou_3d_mean
        dict_wandb["test_iou_img"] = iou_img_mean
    return dict_wandb


if __name__ == '__main__':
    opt = config_parser()
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    opt.device = device

    main()
