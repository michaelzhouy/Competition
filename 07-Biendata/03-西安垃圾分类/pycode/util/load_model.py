# -*- coding: utf-8 -*-
# @Time    : 2021/9/23 9:51 上午
# @Author  : Michael Zhouy
import torch
from torch import nn
import timm
import torch.nn.functional as F
# from torch.cuda.amp import autocast
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet50
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5,6,7'


def load_model(net, num_classes):
    num_classes = num_classes
    if net == 'efficientnet-b0':
        net = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
    if net == 'efficientnet-b1':
        net = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
    if net == 'efficientnet-b2':
        net = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
    if net == 'efficientnet-b3':
        net = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
    elif net == 'efficientnet-b4':
        net = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
    elif net == 'efficientnet-b5':
        net = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)
    elif net == 'efficientnet-b6':
        net = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)
    elif net == 'efficientnet-b7':
        net = EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)
    elif net == 'resnet50':
        net = resnet50(pretrained=True)
        # net.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    elif net == 'swin_base_patch4_window7_224':
        net = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)
    elif net == 'swin_large_patch4_window7_224':
        net = timm.create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=num_classes)
    elif net == 'swin_base_patch4_window12_384':
        net = timm.create_model('swin_base_patch4_window12_384', pretrained=True, num_classes=num_classes)
    elif net == 'swin_large_patch4_window12_384':
        net = timm.create_model('swin_large_patch4_window12_384', pretrained=True, num_classes=num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 训练使用多GPU，测试单GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net = net.to(device)
    # print(net)
    return net
