###########################################################################
# Referred to: https://github.com/zhanghang1989/PyTorch-Encoding
###########################################################################
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.scatter_gather import scatter
import threading
import torch
from torch.cuda._utils import _get_device_index
from torch.cuda.amp import autocast
from torch._utils import ExceptionWrapper

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

__all__ = ['MultiEvalModule']

class MultiEvalModule(DataParallel):
    """Multi-size Segmentation Eavluator"""
    def __init__(self, module, nclass, device_ids=None, flip=True,
                 scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]):
                # scales=[1.0]):
        super(MultiEvalModule, self).__init__(module, device_ids)
        self.nclass = nclass
        self.base_size = module.base_size
        self.crop_size = module.crop_size
        self.scales = scales
        self.flip = flip
        print('MultiEvalModule: base_size {}, crop_size {}'. \
            format(self.base_size, self.crop_size))

    def parallel_forward(self, inputs, **kwargs):
        """Multi-GPU Mult-size Evaluation

        Args:
            inputs: list of Tensors
        """
        # print('additional_utils/encoding_models.py, parallel_forward function, where does the inputs come from ', len(inputs))
        inputs = [(input.unsqueeze(0).cuda(device),)
                  for input, device in zip(inputs, self.device_ids)]
        # 将模型复制到目标GPU上。replicate函数用于创建模型的多个副本，每个副本放在不同的GPU上。
        replicas = self.replicate(self, self.device_ids[:len(inputs)])
        # 如果kwargs不为空，将其分散到目标GPU上。
        kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        #for out in outputs:
        #    print('out.size()', out.size())

        return outputs

    def forward(self, image):
        """Mult-size Evaluation"""
        # only single image is supported for evaluation
        # print('additional_utils/encoding_models.py, forward , input image is ', image.shape)
        batch, _, h, w = image.size()
        # _, h, w = image.size()
        # image = image.unsqueeze(0).cuda(0)
        assert(batch == 1)

        stride_rate = 2.0/3.0
        # crop_size = self.crop_size
        crop_size = 512
        # print('the input crop_size is ', crop_size)
        stride = int(crop_size * stride_rate) # stride = 320
        
        # 在图像所在的设备上，创建(batch,self.nclass,h,w)张量，并初始化为0
        # image.new(), 创建一个与 image 的类型和设备相同的新张量,
        with torch.cuda.device_of(image):
            scores = image.new().resize_(batch,self.nclass,h,w).zero_().cuda()

        self.scales=[1.0]
        # 遍历所有的尺寸
        for scale in self.scales:
            # for scale=1.0
            # print('start iterate scale ', scale,', totally 6 scale')
            long_size = int(math.ceil(self.base_size * scale)) # base size = 520, long_size = 520
            if h > w: # h=480,w=640,false 
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
                short_size = width
            else: # this branch
                width = long_size # width = 520
                height = int(1.0 * h * long_size / w + 0.5) # height=390
                short_size = height # short_size=390
            """
            short_size = int(math.ceil(self.base_size * scale))
            if h > w:
                width = short_size
                height = int(1.0 * h * short_size / w)
                long_size = height
            else:
                height = short_size
                width = int(1.0 * w * short_size / h)
                long_size = width
            """
            # resize image to current size
            # cur_img = resize_image(image, height, width, **self.module._up_kwargs)
            cur_img = image
            # print('img resize to shape ', cur_img.shape)

            outputs = module_inference(self.module, cur_img, self.flip)

        return outputs


def module_inference(module, image, flip=False):
    # 返回预测值output，是一个多通道的，（bsz,nclass,h,w）
    # print('original forward')
    output = module.evaluate(image)
    if flip:
        fimg = flip_image(image)
        # 预测翻转图像的pred结果
        print('flip forward')
        foutput = module.evaluate(fimg)
        # 原始结果+翻转结果，数值相加
        output += flip_image(foutput)
    return output

def resize_image(img, h, w, **up_kwargs):
    return F.interpolate(img, (h, w), **up_kwargs)

def pad_image(img, mean, std, crop_size):
    b,c,h,w = img.size()
    assert(c==3)
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b,c,h+padh,w+padw)
    for i in range(c):
        # note that pytorch pad params is in reversed orders
        img_pad[:,i,:,:] = F.pad(img[:,i,:,:], (0, padw, 0, padh), value=pad_values[i])
    assert(img_pad.size(2)>=crop_size and img_pad.size(3)>=crop_size)
    return img_pad

def crop_image(img, h0, h1, w0, w1):
    return img[:,:,h0:h1,w0:w1]

def flip_image(img):
    assert(img.dim()==4) # 输入必须是4通道的（bsz,nclass,h,w）
    with torch.cuda.device_of(img):
        # img.size(3) 是图像的宽度w，
        # idx，倒序的向量
        idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
    return img.index_select(3, idx)
    # 使用索引选择沿着宽度维度（第4维）翻转图像。idx 是倒序的索引，因此 index_select 会按照从右到左的顺序重新排列宽度维度上的像素，从而实现水平翻转。