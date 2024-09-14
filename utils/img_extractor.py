import ipdb
import torch
import torch.nn as nn
import torchvision


class Global_Img_Extractor(nn.Module):
    def __init__(self, opt):
        super(Global_Img_Extractor, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=False)
        resnet.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.linear = nn.Linear(1000, opt.brick_feature_len)
        self.img_feature = resnet

    def forward(self, imgs):
        global_img_f = []
        for view in range(imgs.shape[1]):
            global_img_f.append(self.img_feature(imgs[:, view:view+1, :, :])[:, None, :])
        global_img_f = torch.cat(global_img_f, dim=1)
        return self.linear(global_img_f)


class Late_Img_Extractor(nn.Module):
    def __init__(self, opt):
        super(Late_Img_Extractor, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=False)
        resnet_extractor = torch.nn.Sequential(*(list(resnet.children())[:-4]))
        resnet_extractor[0]= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        conv = nn.Conv2d(128, opt.img_f_num, kernel_size=3, stride=1, padding=1,bias=False)
        resnet_extractor.add_module('conv', conv)
        resnet_extractor.add_module('bn', nn.BatchNorm2d(opt.img_f_num))
        self.img_feature = resnet_extractor

    def forward(self, imgs):
        feature = []
        for view in range(imgs.shape[1]):
            feature.append(self.img_feature(imgs[:, view:view+1, :, :]))
        feature = torch.cat(feature, dim=1)
        return feature.reshape(feature.shape[0], feature.shape[1], -1)


class Early_Img_Extractor(nn.Module):
    def __init__(self, opt):
        super(Early_Img_Extractor, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=False)
        resnet_extractor = torch.nn.Sequential(*(list(resnet.children())[:-4]))
        resnet_extractor[0]= nn.Conv2d(opt.view_num, 64, kernel_size=7, stride=2, padding=3,bias=False)
        conv = nn.Conv2d(128, opt.img_f_num, kernel_size=3, stride=1, padding=1,bias=False)
        resnet_extractor.add_module('conv', conv)
        resnet_extractor.add_module('bn', nn.BatchNorm2d(opt.img_f_num))
        self.img_feature = resnet_extractor

    def forward(self, imgs):
        feature = self.img_feature(imgs)
        return feature.reshape(feature.shape[0], feature.shape[1], -1)
