import re
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from argparse import ArgumentParser
import pytorch_lightning as pl
from .lsegmentation_module import LSegmentationModule
from .models.lseg_net import LSegNet
from encoding.models.sseg.base import up_kwargs

import os
import clip
import numpy as np

from scipy import signal
import glob

from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# LSegModule 类继承自 LSegmentationModule 类
class LSegModule(LSegmentationModule):
    # 数据路径，数据集，batchsize等信息输 
    def __init__(self, data_path, dataset, batch_size, base_lr, max_epochs, **kwargs):

        # 方法接收了多个参数， 并将这些参数传递给父类 (LSegmentationModule) 的构造方(super().__init__())
        super(LSegModule, self).__init__(
            data_path, dataset, batch_size, base_lr, max_epochs, **kwargs
        )

        if dataset == "citys":
            self.base_size = 2048
            self.crop_size = 768
        else:
            self.base_size = 1024
            self.crop_size = 1024

        # 使用预训练模
        use_pretrained = True

        norm_mean= [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]

        print('modules/lseg_module.py, ** Use norm {}, {} as the mean and std **'.format(norm_mean, norm_std))

        train_transform = [
            transforms.ToTensor(), # 图像数据转换为张量格  
            transforms.Normalize(norm_mean, norm_std),# 对张量进行归一化处     
        ]

        val_transform = [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std), # 对张量进行归一化处     
        ]

        self.train_transform = transforms.Compose(train_transform) # 将多个数据转换操作组合成一个操作序�?        
        self.val_transform = transforms.Compose(val_transform)
        
        # 用于获取训练集和验证集  
        self.trainset = self.get_trainset(
            dataset, # 数据集名
            augment=kwargs["augment"], # false
            base_size=self.base_size, # 基础尺寸           
            crop_size=self.crop_size, # 裁剪尺寸
        )
        
        self.valset = self.get_valset(
            dataset,
            augment=kwargs["augment"], # false
            base_size=self.base_size,
            crop_size=self.crop_size,
        )

        use_batchnorm = (
            (not kwargs["no_batchnorm"]) if "no_batchnorm" in kwargs else True
        )
        # print(kwargs)

        labels = self.get_labels('buff4k')

        # 创建了一个 LSegNet 的实例 self.net。       
        self.net = LSegNet(
            labels=labels, # 获取了buff6k的labels文本
            backbone=kwargs["backbone"],
            features=kwargs["num_features"], # 特征维度
            crop_size=self.crop_size,
            arch_option=kwargs["arch_option"],
            block_depth=kwargs["block_depth"],
            activation=kwargs["activation"],
        )

        # 设置预训练模型参数      
        self.net.pretrained.model.patch_embed.img_size = (
            self.crop_size,
            self.crop_size,
        )

        self._up_kwargs = up_kwargs
        self.mean = norm_mean
        self.std = norm_std

        self.criterion = self.get_criterion(**kwargs)

    def get_labels(self, dataset):
        print('modules/lseg_module.py, get_labels funciton, starting "get_labels" function ')
        labels = []
        path = 'label_files/{}_objectInfo.txt'.format(dataset)
        
        assert os.path.exists(path), '*** Error : {} not exist !!!'.format(path)
        f = open(path, 'r') 
        lines = f.readlines() # 读取文件的所有行，并将其存储在列表lines中   
        for line in lines: 
            label = line.strip().split(',')[-1].split(';')[0] # 然后按逗号 , 分割字符串，并取最后一个分割后的部分。
            labels.append(label)
        f.close()
        if dataset in ['buff4k']:
            labels = labels[1:] # 排除列表中的第一个元素，可能是文件的标题行或其他无效数据。     
            print('modules/lseg_module.py, get_labels funciton, buff4k get_labels result: ', labels)
        return labels


    @staticmethod # 装饰器表示 add_model_specific_args 是一个静态方法，可以直接通过类调用，而不需要先创建类的实例。    
    def add_model_specific_args(parent_parser):
        parser = LSegmentationModule.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser])

        parser.add_argument(
            "--backbone",
            type=str,
            default="clip_vitl16_384",
            help="backbone network",
        )

        parser.add_argument(
            "--num_features",
            type=int,
            default=256,
            help="number of featurs that go from encoder to decoder",
        )

        parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")

        parser.add_argument(
            "--finetune_weights", type=str, help="load weights to finetune from"
        )

        parser.add_argument(
            "--no-scaleinv",
            default=True,
            action="store_false",
            help="turn off scaleinv layers",
        )

        parser.add_argument(
            "--no-batchnorm",
            default=False,
            action="store_true",
            help="turn off batchnorm",
        )

        parser.add_argument(
            "--widehead", default=False, action="store_true", help="wider output head"
        )

        parser.add_argument(
            "--widehead_hr",
            default=False,
            action="store_true",
            help="wider output head",
        )

        parser.add_argument(
            "--arch_option",
            type=int,
            default=0,
            help="which kind of architecture to be used",
        )

        parser.add_argument(
            "--block_depth",
            type=int,
            default=0,
            help="how many blocks should be used",
        )

        parser.add_argument(
            "--activation",
            choices=['lrelu', 'tanh'],
            default="lrelu",
            help="use which activation to activate the block",
        )

        return parser
