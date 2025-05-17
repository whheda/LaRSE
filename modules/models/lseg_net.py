import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lseg_blocks import FeatureFusionBlock, Interpolate, _make_encoder, FeatureFusionBlock_custom, forward_vit
import clip
import numpy as np
import pandas as pd
import os

class depthwise_clipseg_conv(nn.Module):
    def __init__(self):
        super(depthwise_clipseg_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    
    def depthwise_clipseg(self, x, channels):
        x = torch.cat([self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)], dim=1)
        return x

    def forward(self, x):
        channels = x.shape[1]
        out = self.depthwise_clipseg(x, channels)
        return out


class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x


class depthwise_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()


    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x

class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)

def _make_fusion_block(features, use_bn):
    # 这里没有传入具体数据
    return FeatureFusionBlock_custom(
        features, # 通道数
        activation=nn.ReLU(False), 
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

# 该类继承自 BaseModel,使用了 CLIP 模型作为其骨干网络，并对其进行了特定的修改以适应图像分割任务的需求
class LSeg(BaseModel):
    def __init__(
        self,
        head, # 输出层，通常是一个卷积层，用于生成最终的分割图。
        features=256,
        backbone="clip_vitl16_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(LSeg, self).__init__()

        self.channels_last = channels_last
        
        # 从网络中提取特定层的特征
        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
        }

        # Instantiate backbone and reassemble blocks，实例化backbone，并重新组装模块。
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features, # 256
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )
        # 创建一个特征融合模块
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        if backbone in ["clipRN50x16_vitl16_384"]:
            self.out_c = 768
        else:
            self.out_c = 512
        
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1) # 一个 1x1 的卷积层，用于调整特征的通道数

        self.arch_option = kwargs["arch_option"] # arch_option=0,
        if self.arch_option == 1:
            self.scratch.head_block = bottleneck_block(activation=kwargs["activation"])
            self.block_depth = kwargs['block_depth']
        elif self.arch_option == 2:
            self.scratch.head_block = depthwise_block(activation=kwargs["activation"])
            self.block_depth = kwargs['block_depth']
        
        # 最终的输出卷积层
        # head是一个双线性插值层
        self.scratch.output_conv = head

        # 将标签进行 token 化处理，这里的self.labels是从 lseg_module
        # print('modules/models/lseg_net.py, self.labels comes from get_label() in lseg_module.py ', len(self.labels))
        self.text = clip.tokenize(self.labels)    
    
    # 输入影像是x，输入text是self.labels
    def forward(self, x, labelset=''):
        if labelset == '':
            # print('modules/models/lseg_net.py, LSeg forward, the labelset is not inputed')
            text = self.text # 执行词元化，将输入的self.labels转换为token
        else:
            text = clip.tokenize(labelset)    
        # print('modules/models/lseg_net.py, LSeg forward, the tokenized text shape', text.shape) # (150x77)

        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)
        
        # 用vit模型，提取特征
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn) # [1, 512, 256, 256], 下采样2倍
        # print('In lseg_net.py, LSeg forward, path_1 feature shape', path_1.shape) 

        text = text.to(x.device)
        self.logit_scale = self.logit_scale.to(x.device)

        # 将文本标签编码为特征
        text_features = self.clip_pretrained.encode_text(text) # [12, 512]，150个单词的token
        # print('In lseg_net.py, LSeg forward, text_features shape', text_features.shape) 
        
        # 使用卷积层 head1 处理图像特征 path_1
        image_features = self.scratch.head1(path_1) # [1, 512, 256, 256]，下采样2倍
        image_features_backup = image_features
        # print('In lseg_net.py, LSeg forward, image_features shape', image_features.shape) 
        image_features_backup = image_features_backup / image_features_backup.norm(dim=-1, keepdim=True)

        imshape = image_features.shape
        image_features = image_features.permute(0,2,3,1).reshape(-1, self.out_c) # [65512, 512], 将每个像素看作是一个token
        # print('In lseg_net.py, LSeg forward, image_features shape after permute', image_features.shape) 

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 计算图像特征与文本特征的点积 logits_per_image
        # print(image_features.dtype)  # 查看 image_features 的数据类型
        # print(text_features.dtype)   # 查看 text_features 的数据类型
        # print(self.logit_scale.dtype)  # 查看 logit_scale 的数据类型
        logits_per_image = self.logit_scale * image_features.half() @ text_features.t().half() # [57600, 150]
        # print('In lseg_net.py, LSeg forward, logits_per_image shape', logits_per_image.shape) 

        # 并调整形状
        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2) # [1, 150, 240, 240]
        # print('In lseg_net.py, LSeg forward, logits_per_image shape after reshape', out.shape) 
        
        # print('self.arch_option ', self.arch_option) 
        if self.arch_option in [1, 2]: # self.arch_option = 0
            for _ in range(self.block_depth - 1):
                out = self.scratch.head_block(out)
            out = self.scratch.head_block(out, False)
        
        # 使用输出卷积层 output_conv 生成最终的分割图
        out = self.scratch.output_conv(out) # [1, 150, 480, 480]
        # print('In lseg_net.py, LSeg forward, out shape ', out.shape)

        return out

# 这个类仅传入了labels，没有image
class LSegNet(LSeg):
    """Network for semantic segmentation."""
    def __init__(self, labels, path=None, scale_factor=0.5, crop_size=480, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256
        kwargs["use_bn"] = True

        self.crop_size = crop_size
        self.scale_factor = scale_factor

        # self.labels是这里传进来的
        self.labels = labels
        
        # head是一个双线性插值层
        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)


    
        
    