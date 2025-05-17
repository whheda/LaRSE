import os
import pathlib

from glob import glob

from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
import numpy as np
import cv2
import random
import math
from torchvision import transforms

# args, LSegModule
def do_training(hparams, model_constructor):
    # instantiate model
    model = model_constructor(**vars(hparams))
    # set all sorts of training parameters
    hparams.gpus = -1
    hparams.accelerator = "ddp" # 分布式数据并行（DDP, Distributed Data Parallel）
    hparams.benchmark = True
    
    # dry run 试运行，执行试运行的一种方法是，使用打印语句查看脚本在每个步骤中会执行的操作，而无需实际执行任何命令
    # dry run命令是从哪里传进来的
    if hparams.dry_run:
        print("Doing a dry run")
        hparams.overfit_batches = hparams.batch_size
    
    # what is it?
    if not hparams.no_resume:
        hparams = set_resume_parameters(hparams)

    if not hasattr(hparams, "version") or hparams.version is None:
        hparams.version = 0

    hparams.sync_batchnorm = True
    
    # save_dir = checkpoints, 日志记录类，用于记录训练过程中的各种信息，比如损失值、指标值、超参数等。
    ttlogger = pl.loggers.TestTubeLogger(
        "checkpoints", name=hparams.exp_name, version=hparams.version
    )
    
    # 返回的是类？
    hparams.callbacks = make_checkpoint_callbacks(hparams.exp_name, hparams.version)
    
    # 创建一个WandbLogger对象，将训练过程中的日志数据记录到Weights and Biases（WandB）平台
    wblogger = get_wandb_logger(hparams)
    # 路径 exp_dir = f"checkpoints/{hparams.exp_name}/version_{hparams.version}/"， 文件名f"{exp_dir}/wandb_id"
    hparams.logger = [wblogger, ttlogger]

    trainer = pl.Trainer.from_argparse_args(hparams)

    # 主要在这个函数里面跑模型
    trainer.fit(model)
    

def get_default_argument_parser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="number of nodes for distributed training",
    )

    parser.add_argument(
        "--exp_name", type=str, required=True, help="name your experiment"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="run on batch of train/val/test",
    )

    parser.add_argument(
        "--no_resume",
        action="store_true",
        default=False, # 默认是，如果文件夹中有checkpoint，则续上，接着训练
        help="resume if we have a checkpoint",
    )

    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="accumulate N batches for gradient computation",
    )

    parser.add_argument(
        "--max_epochs", type=int, default=200, help="maximum number of epochs"
    )

    parser.add_argument(
        "--project_name", type=str, default="lightseg", help="project name for logging"
    )

    return parser


def make_checkpoint_callbacks(exp_name, version, base_path="checkpoints", frequency=1): # frequency，保存checkpoint的频率
    version = 0 if version is None else version

    base_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{base_path}/{exp_name}/version_{version}/checkpoints/",
        save_last=True,
        verbose=True, # 设置为True，将打印有关检查点保存的信息
    )

    val_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc_epoch", # 监控验证集上的指标val_acc_epoch
        dirpath=f"{base_path}/{exp_name}/version_{version}/checkpoints/",
        filename="result-{epoch}-{val_acc_epoch:.2f}", # 保存checkpoint的名字
        mode="max", 
        save_top_k=3, # 保存验证集指标最高的k个checkpoint
        verbose=True, # 打印有关checkpoint保存的信息
    )

    return [base_callback, val_callback]


def get_latest_version(folder):
    versions = [
        int(pathlib.PurePath(path).name.split("_")[-1])
        for path in glob(f"{folder}/version_*/")
    ]

    if len(versions) == 0:
        return None

    versions.sort()
    return versions[-1]

# 实验名称exp_name，版本号version
# 函数目的：模型训练或恢复过程中，找到最新的检查点文件以继续训练或进行评估
def get_latest_checkpoint(exp_name, version):
    while version > -1:
        # 构建checkpoint路径
        folder = f"./checkpoints/{exp_name}/version_{version}/checkpoints/"

        # checkpoint路径
        latest = f"{folder}/last.ckpt"
        # 
        if os.path.exists(latest):
            return latest, version
            # 如果文件存在，则返回该路径和当前版本号

        # 如果不存在
        # 查找匹配模式{folder}/epoch=*.ckpt的所有checkpoint文件，并将结果存储在列表chkpts中
        chkpts = glob(f"{folder}/epoch=*.ckpt")
        
        # 如果 如果找到的检查点文件列表chkpts不为空，则跳出循环。
        if len(chkpts) > 0:
            break

        version -= 1
    # 如果没有找到任何检查点文件，返回None, None。
    if len(chkpts) == 0:
        return None, None

    # 从找到的检查点文件列表中，选择最新的文件
    latest = max(chkpts, key=os.path.getctime)

    return latest, version

# 
def set_resume_parameters(hparams):
    version = get_latest_version(f"./checkpoints/{hparams.exp_name}")
    print('utils.py, set_resume_parameters function, the latest_version is ', version)

    if version is not None:
        # 指定实验名称和版本号的最新检查点文件路径和版本
        latest, version = get_latest_checkpoint(hparams.exp_name, version)
        print(f"Resuming checkpoint {latest}, exp_version={version}")
        
        # 设置hparams对象的resume_from_checkpoint属性，为最新的checkpoint文件路径
        hparams.resume_from_checkpoint = latest 
        hparams.version = version

        wandb_file = "checkpoints/{hparams.exp_name}/version_{version}/wandb_id"
        if os.path.exists(wandb_file):
            with open(wandb_file, "r") as f:
                # 如果该文件存在，读取文件内容并将其赋值给hparams对象的wandb_id属性
                hparams.wandb_id = f.read() # 权重和参数存储在wandb_id里面
    else:
        version = 0

    return hparams

# weight and biase
def get_wandb_logger(hparams):
    # 实验的目录路径checkpoints/exp_name/version_version/
    exp_dir = f"checkpoints/{hparams.exp_name}/version_{hparams.version}/"
    # 保存WandB运行ID的文件路径
    id_file = f"{exp_dir}/wandb_id"

    if os.path.exists(id_file):
        # 如果id_file存在，从文件中读取WandB,并赋值给hparams.wandb_id
        with open(id_file) as f:
            hparams.wandb_id = f.read()
    else:
        hparams.wandb_id = None
    
    # 创建WandbLogger对象
    logger = pl.loggers.WandbLogger(
        save_dir="checkpoints", # 日志保存的目录
        project=hparams.project_name,
        name=hparams.exp_name,
        id=hparams.wandb_id,
    )
    
    # 强制初始化logger.experiment
    if hparams.wandb_id is None:
        _ = logger.experiment
    
    # 如果不存在实验目录的路径
    if not os.path.exists(exp_dir):
        # 创建路径
        os.makedirs(exp_dir)
        # 将logger.version写入id_file
        with open(id_file, "w") as f:
            f.write(logger.version)

    return logger


class Resize(object):
    """Resize sample to given size (width, height)."""

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
        letter_box=False,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional): # 长宽比
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method
        self.__letter_box = letter_box
    
    # 确保数值x被调整为self.__multiple_of的倍数，并在必要时进行范围约束
    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        # 将输入的x调整为self.__multiple_of的倍数。
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)
        # 如果调整后的数值y大于max_val，则向下取最近的倍数
        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)
        # 如果调整后的数值y小于min_val，则向上取最近的倍数。
        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):

        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal": # 尽可能少地调整比例
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )
        # 调整后的高度和宽度使用constrain_to_multiple_of函数进行约束
        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    # 将输入图像sample进行填充，使其尺寸变为self.__height和self.__width，保持原图像的中心位置不变，周围填充常数值（默认为黑色）
    def make_letter_box(self, sample):
        top = bottom = (self.__height - sample.shape[0]) // 2
        left = right = (self.__width - sample.shape[1]) // 2
        sample = cv2.copyMakeBorder(
            sample, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0
        )
        return sample

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )
        print('utils.py, image w and h before resize, ', sample["image"].shape[1], ' ', sample["image"].shape[0])
        print('utils.py, image w and h after resize, ', width, ' ', height)

        # resize image
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )
        
        print('utils.py, image before __letter_box', sample["image"].shape[1], ' ', sample["image"].shape[0])
        if self.__letter_box:
            sample["image"] = self.make_letter_box(sample["image"])
        print('utils.py, image after __letter_box', sample["image"].shape[1], ' ', sample["image"].shape[0])

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

                if self.__letter_box:
                    sample["disparity"] = self.make_letter_box(sample["disparity"])

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

                if self.__letter_box:
                    sample["depth"] = self.make_letter_box(sample["depth"])
            
            # resize mask
            print('utils.py, mask w and h before resize, ', sample["image"].shape[1], ' ', sample["image"].shape[0])
            print('utils.py, mask w and h after resize, ', width, ' ', height)
            sample["mask"] = cv2.resize(
                sample["mask"].astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

            print('utils.py, image before __letter_box', sample["image"].shape[1], ' ', sample["image"].shape[0])
            
            if self.__letter_box:
                sample["mask"] = self.make_letter_box(sample["mask"])
            print('utils.py, image after __letter_box', sample["image"].shape[1], ' ', sample["image"].shape[0])

            sample["mask"] = sample["mask"].astype(bool)

        return sample
