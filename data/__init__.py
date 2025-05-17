import copy

import itertools
import functools
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as torch_transforms
import encoding.datasets as enc_ds


# 创建字典 encoding_datasets，
# 键是数据集名称（如 "coco"、"ade20k" 等），
# 值是enc_ds.get_dataset
# functools.partial函数直接运行enc_ds.get_dataset，其中传参是x
encoding_datasets = {
    x: functools.partial(enc_ds.get_dataset, x)
    for x in ["coco","buff1w","buff6k","buff4k","ade20k", "pascal_voc", "pascal_aug", "pcontext", "citys"]
} # 这些并不是名字，而是函数

# 传递进来的参数有（**kwargs）：
# root：数据集的根目录
# split：数据集的划分（例如 train、test 等）
# mode：数据集的模式，通常用于指定数据预处理或增强的方式
# transform： 图像数据转换为张量格式，归一化操作

def get_dataset(name, **kwargs):
    if name in encoding_datasets:
        return encoding_datasets[name.lower()](**kwargs)
    assert False, f"dataset {name} not found"


def get_available_datasets():
    return list(encoding_datasets.keys()) 
