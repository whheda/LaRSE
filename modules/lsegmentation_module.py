import types
import time
import random
import clip
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from argparse import ArgumentParser

import pytorch_lightning as pl

from data import get_dataset, get_available_datasets

from encoding.models import get_segmentation_model
from encoding.nn import SegmentationLosses

from encoding.utils import batch_pix_accuracy, batch_intersection_union

# add mixed precision
import torch.cuda.amp as amp
import numpy as np

from encoding.utils import SegmentationMetric

class LSegmentationModule(pl.LightningModule): # 基于 PyTorch Lightning 框架的模型训练模块，专门用于图像分割任务
# 调用父类 pl.LightningModule 的初始化方法，确保正确初始化继承LightningModule 的功
    def __init__(self, data_path, dataset, batch_size, base_lr, max_epochs, **kwargs):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.base_lr = base_lr / 16 * batch_size
        self.lr = self.base_lr

        self.epochs = max_epochs
        self.other_kwargs = kwargs
        self.enabled = False #True mixed precision will make things complicated and leading to NAN error
        self.scaler = amp.GradScaler(enabled=self.enabled)

    def forward(self, x):
        return self.net(x) # lseg_module.py中定义的self.net, 是LSegNet类的实例，LSegNet在lseg_net.py中定
    
    # 如果没有target传参，会在“return pred”位置停止
    def evaluate(self, x, target=None):
        pred = self.net.forward(x)
        # print('modules/lsegmentation_module.py, evaluate function, pred shape ', image_feature.shape)
        if isinstance(pred, (tuple, list)):
            # 检查预测结果是否为元组或列表类型，如果是，则取第一个元素        
            pred = pred[0]
        # 如果没有提供label，则直接返回预测结果 pred
        # pred 是一个多通道的，（bsz,nclass,h,w）
        if target is None:
            return pred
        # return pixel_correct, pixel_labeled   
        # 这里得到的就是所有标签的像素点的总数，和所有预测正确的像素点的总数
        # batch_pix_accuracy在encoding/utils/metrics.py里面定义
        # correct：预测正确的像素数量，labeled：总的像素数量
        # 这个函数没有分类      
        correct, labeled = batch_pix_accuracy(pred.data, target.data) # target是本函数的传参，

        # inter为向量，长度nclass，代表每个类预测正确的像素数量；
        # union为向量，长度nclass，代表所有预测出来的每个类的像素数量
        print('lsegmentation_module.py, evaluate function, self.nclass', self.nclass)    
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
        print('modules/lsegmentation_module.py, evaluate function, correct, labeled, inter, union ', correct, labeled, inter, union)
        return correct, labeled, inter, union

    def evaluate_random(self, x, labelset, target=None):
        pred = self.net.forward(x, labelset)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)

        return correct, labeled, inter, union
    

    def training_step(self, batch, batch_nb):
        img, target, name = batch
        print('modules/lsegmentation_module.py, training_step, img', img.shape)
        print('modules/lsegmentation_module.py, training_step, target', target.shape)
        print('modules/lsegmentation_module.py, training_step, name', name)

        with amp.autocast(enabled=self.enabled):# 是否启用混合精度训练
            out = self(img) # self(img) 为执行forward，得到输出out, forward调用self.net,

            # self.net的forward的输出是特征图，为何这里却是loss函数
            print('modules/lsegmentation_module.py, training_step, out shape', out[0].shape) # 10x1024x1024
            # self.net在LSegementationModule的子类LSegModule中创建，并调用LSegNet类，在LSegNet类中具体定义了forward内容
            multi_loss = isinstance(out, tuple)  # 检查是否有多个输出
            if multi_loss:
                print('lsegmenttation_module.py, training_step, multi_loss')
                loss = self.criterion(*out, target)
            else:
                print('lsegmenttation_module.py, training_step, single_loss')
                loss = self.criterion(out, target)
            loss = self.scaler.scale(loss) # 使用 self.scaler 对损失进行缩放，用于混合精度训练
        # 获取最终的分类结果final_output
        final_output = out[0] if multi_loss else out

        # _filter_invalid只保留有效面积进行精度评     
        train_pred, train_gt = self._filter_invalid(final_output, target)
        if train_gt.nelement() != 0:
            self.train_accuracy(train_pred, train_gt)
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc_epoch", self.train_accuracy.compute())

    def validation_step(self, batch, batch_nb):
        img, target, name = batch

        print('modules/lsegmentation_module.py, validation_step, img', img.shape)
        print('modules/lsegmentation_module.py, validation_step, target', target.shape)
        print('modules/lsegmentation_module.py, validation_step, name', name)

        out = self(img) 
        multi_loss = isinstance(out, tuple)
        if multi_loss:
            val_loss = self.criterion(*out, target)
        else:
            val_loss = self.criterion(out, target)
        final_output = out[0] if multi_loss else out
        valid_pred, valid_gt = self._filter_invalid(final_output, target)
        self.val_iou.update(target, final_output)
        pixAcc, iou = self.val_iou.get()
        self.log("val_loss_step", val_loss)
        self.log("pix_acc_step", pixAcc)
        self.log(
            "val_acc_step",
            self.val_accuracy(valid_pred, valid_gt),
        )
        self.log("val_iou", iou)

    def validation_epoch_end(self, outs):
        pixAcc, iou = self.val_iou.get()
        self.log("val_acc_epoch", self.val_accuracy.compute())
        self.log("val_iou_epoch", iou)
        self.log("pix_acc_epoch", pixAcc)

        self.val_iou.reset()

    def _filter_invalid(self, pred, target):
        valid = target != self.other_kwargs["ignore_index"]
        _, mx = torch.max(pred, dim=1)
        return mx[valid], target[valid]

    def configure_optimizers(self):
        params_list = [
            {"params": self.net.pretrained.parameters(), "lr": self.base_lr},
        ]
        if hasattr(self.net, "scratch"):
            print("Found output scratch")
            params_list.append(
                {"params": self.net.scratch.parameters(), "lr": self.base_lr * 10}
            )
        if hasattr(self.net, "auxlayer"):
            print("Found auxlayer")
            params_list.append(
                {"params": self.net.auxlayer.parameters(), "lr": self.base_lr * 10}
            )
        if hasattr(self.net, "scale_inv_conv"):
            print(self.net.scale_inv_conv)
            print("Found scaleinv layers")
            params_list.append(
                {
                    "params": self.net.scale_inv_conv.parameters(),
                    "lr": self.base_lr * 10,
                }
            )
            params_list.append(
                {"params": self.net.scale2_conv.parameters(), "lr": self.base_lr * 10}
            )
            params_list.append(
                {"params": self.net.scale3_conv.parameters(), "lr": self.base_lr * 10}
            )
            params_list.append(
                {"params": self.net.scale4_conv.parameters(), "lr": self.base_lr * 10}
            )

        if self.other_kwargs["midasproto"]:
            print("Using midas optimization protocol")
            
            opt = torch.optim.Adam(
                params_list,
                lr=self.base_lr,
                betas=(0.9, 0.999),
                weight_decay=self.other_kwargs["weight_decay"],
            )
            sch = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda x: pow(1.0 - x / self.epochs, 0.9)
            )

        else:
            opt = torch.optim.SGD(
                params_list,
                lr=self.base_lr,
                momentum=0.9,
                weight_decay=self.other_kwargs["weight_decay"],
            )
            sch = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda x: pow(1.0 - x / self.epochs, 0.9)
            )
        return [opt], [sch]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
            worker_init_fn=lambda x: random.seed(time.time() + x),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
        )

    def get_trainset(self, dset, augment=False, **kwargs):
        print(kwargs)
        if augment == True:
            mode = "train_x"
        else: # augment是False，选这          
            mode = "test" # train, test

        print('modules/lsegmentation_module.py, get into train mode ', mode)
        
        # 将数据路径，模式等参数传进去encoding文件夹的ade20k.py
        dset = get_dataset(
            dset,
            root=self.data_path,
            split="train",
            mode=mode,
            transform=self.train_transform, # 图像数据转换为张量格式，归一化操作       
            **kwargs
        )
        
        self.num_classes = dset.num_class
        print('lsegmentation_module.py, get_trainset, self.num_classes ', self.num_classes)
        self.train_accuracy = pl.metrics.Accuracy()

        return dset

    def get_valset(self, dset, augment=False, **kwargs):
        self.val_accuracy = pl.metrics.Accuracy()
        self.val_iou = SegmentationMetric(self.num_classes)

        if augment == True:
            mode = "val_x"
        else: # augment是False，选这       
            mode = "val"

        print('modules/lsegmentation_module.py, get into val mode ', mode)
        
        # 判断给定数据名称是否在数据集列表中，如果在，则返回名    
        return get_dataset(
            dset,
            root=self.data_path,
            split="val",
            mode=mode,
            transform=self.val_transform,
            **kwargs
        )


    def get_criterion(self, **kwargs):
        print('lsegmentation_module.py, get_criterion, kwargs["se_loss"] ', kwargs["se_loss"])
        print('lsegmentation_module.py, get_criterion, kwargs["aux"] ', kwargs["aux"])
        print('lsegmentation_module.py, get_criterion, kwargs["se_weight"] ', kwargs["se_weight"])
        print('lsegmentation_module.py, get_criterion, kwargs["aux_weight"] ', kwargs["aux_weight"])
        print('lsegmentation_module.py, get_criterion, kwargs["ignore_index"] ', kwargs["ignore_index"])
        
        return SegmentationLosses(
            se_loss=kwargs["se_loss"], 
            aux=kwargs["aux"], 
            nclass=self.num_classes, 
            se_weight=kwargs["se_weight"], 
            aux_weight=kwargs["aux_weight"], 
            ignore_index=kwargs["ignore_index"], 
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--data_path", type=str, help="path where dataset is stored"
        )
        parser.add_argument(
            "--dataset",
            choices=get_available_datasets(),
            default="buff4k",
            help="dataset to train on",
        )
        parser.add_argument(
            "--batch_size", type=int, default=16, help="size of the batches"
        )
        parser.add_argument(
            "--base_lr", type=float, default=0.004, help="learning rate"
        )
        parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
        parser.add_argument(
            "--weight_decay", type=float, default=1e-4, help="weight_decay"
        )
        parser.add_argument(
            "--aux", action="store_true", default=False, help="Auxilary Loss"
        )
        parser.add_argument(
            "--aux-weight",
            type=float,
            default=0.2,
            help="Auxilary loss weight (default: 0.2)",
        )
        parser.add_argument(
            "--se-loss",
            action="store_true",
            default=False,
            help="Semantic Encoding Loss SE-loss",
        )
        parser.add_argument(
            "--se-weight", type=float, default=0.2, help="SE-loss weight (default: 0.2)"
        )

        parser.add_argument(
            "--midasproto", action="store_true", default=False, help="midasprotocol"
        )

        parser.add_argument(
            "--ignore_index",
            type=int,
            default=-1,
            help="numeric value of ignore label in gt",
        )
        parser.add_argument(
            "--augment",
            action="store_true",
            default=False,
            help="Use extended augmentations",
        )

        return parser
