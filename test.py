import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
import encoding.utils as utils
from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import test_batchify_fn 
from encoding.models.sseg import BaseNet
from modules.lseg_module import LSegModule
from utils import Resize
import cv2
import math
import types
import functools
import torchvision.transforms as torch_transforms
import copy
import itertools
from PIL import Image
import matplotlib.pyplot as plt
import clip
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from data import get_dataset
from additional_utils.encoding_models import MultiEvalModule as LSeg_MultiEvalModule
import torchvision.transforms as transforms
from torchinfo import summary
from thop import profile
from fvcore.nn import FlopCountAnalysis
import datetime

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description="PyTorch Segmentation")
        # model and dataset
        parser.add_argument(
            "--model", type=str, default="encnet", help="model name (default: encnet)"
        )
        parser.add_argument(
            "--backbone",
            type=str,
            default="clip_vitl16_384",
            help="backbone name (default: resnet50)",
        )
        # 给定coco dataset就能自动索引到数据路径，coco.py中给了数据路径
        parser.add_argument(
            "--dataset",
            type=str,
            default="buff6k",
            help="dataset name (default: pascal12)",
        )
        parser.add_argument(
            "--workers", type=int, default=16, metavar="N", help="dataloader threads"
        )
        parser.add_argument(
            "--base-size", type=int, default=520, help="base image size"
        )
        parser.add_argument(
            "--crop_size", type=int, default=480, help="crop image size"
        )
        parser.add_argument(
            "--train-split",
            type=str,
            default="train",
            help="dataset train split (default: train)",
        )
        # training hyper params
        parser.add_argument(
            "--aux", action="store_true", default=False, help="Auxilary Loss"
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
            "--batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for \
                            training (default: auto)",
        )
        parser.add_argument(
            "--test-batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for \
                            testing (default: same as batch size)",
        )
        # cuda, seed and logging
        parser.add_argument(
            "--no-cuda",
            action="store_true",
            default=False,
            help="disables CUDA training",
        )
        parser.add_argument(
            "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
        )
        parser.add_argument(
            "--weights", type=str, default=None, help="checkpoint to test"
        )
        parser.add_argument(
            "--eval", action="store_true", default=False, help="evaluating mIoU"
        )
        parser.add_argument(
            "--export",
            type=str,
            default=None, # False?
            help="put the path to resuming file if needed",
        )

        parser.add_argument(
            "--acc-bn",
            action="store_true",
            default=False,
            help="Re-accumulate BN statistics",
        )
        parser.add_argument(
            "--test-val",
            action="store_true",
            default=False,
            help="generate masks on val set",
        )
        parser.add_argument(
            "--no-val",
            action="store_true",
            default=False,
            help="skip validation during training",
        )
        parser.add_argument(
            "--module",
            default='lseg',
            help="select model definition",
        )
        # test option
        # --data-path测试数据集的路径
        parser.add_argument(
            "--data_path", type=str, default=None, help="path to test image folder"
        )
        parser.add_argument(
            "--no-scaleinv",
            dest="scale_inv",
            default=True,
            action="store_false",
            help="turn off scaleinv layers",
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
        # 不参与loss计算的数值
        parser.add_argument(
            "--ignore_index",
            type=int,
            default=-1,
            help="numeric value of ignore label in gt",
        )
        parser.add_argument(
            "--label_src",
            type=str,
            default="default",
            help="how to get the labels",
        )
        parser.add_argument(
            "--jobname",
            type=str,
            default="default",
            help="select which dataset",
        )
        parser.add_argument(
            "--no-strict",
            dest="strict",
            default=True,
            action="store_false",
            help="no-strict copy the model",
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

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args


def test(args):

    # 首先执行这个module！
    module = LSegModule.load_from_checkpoint(
        checkpoint_path=args.weights,
        data_path=args.data_path, # 
        dataset=args.dataset, 
        backbone=args.backbone, # clip_vitl16_384
        aux=args.aux, # 
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=args.ignore_index,
        dropout=0.0,
        scale_inv=args.scale_inv,
        augment=False,
        no_batchnorm=False,
        widehead=args.widehead,
        widehead_hr=args.widehead_hr,
        map_locatin="cpu",
        arch_option=args.arch_option,
        strict=args.strict,
        block_depth=args.block_depth,
        activation=args.activation,
    )
    input_transform = module.val_transform # 
    num_classes = module.num_classes
    print('test_lseg.py, "num_classes" come from lseg_module.py的get_label', num_classes)

    # dataset，
    testset = get_dataset(
        args.dataset, # coco
        root=args.data_path, # 
        split="val",
        mode="testval",
        transform=input_transform, # 转tensor，归一化0.5
    )
    # return img，mask
    # dataset是一个ADE20KSegmentation类的实例对象
    # .images/.masks是图像路径的列表，例如，'/home/heda/zyy/seg/datasets/ADEChallengeData2016/images/validation/ADE_val_00001040.jpg'
    print('test_lseg.py, what is testset img', len(testset.images))
    print('test_lseg.py, what is testset mask', len(testset.masks))
    print('test_lseg.py, what is testset NUM_CLASS ', testset.NUM_CLASS)

    # dataloader
    loader_kwargs = (
        {"num_workers": args.workers, "pin_memory": True} if args.cuda else {} # worker=16
    )
    
    test_data = data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=test_batchify_fn,
        **loader_kwargs
    )
    # test_data是一个DataLoader类型
    print('test_lseg.py, check the test_data',test_data)

    if isinstance(module.net, BaseNet):
        model = module.net
    else:
        model = module

    model = model.eval()
    model = model.cpu()

    print(model)

    # test，acc_bn=false
    if args.acc_bn:
        from encoding.utils.precise_bn import update_bn_stats

        data_kwargs = {
            "transform": input_transform,
            "base_size": args.base_size,
            "crop_size": args.crop_size,
        }
        trainset = get_dataset(
            args.dataset, split=args.train_split, mode="train", **data_kwargs
        )
        trainloader = data.DataLoader(
            ReturnFirstClosure(trainset),
            root=args.data_path,
            batch_size=args.batch_size,
            drop_last=True,
            shuffle=True,
            **loader_kwargs
        )
        # print("test_lseg.py, Reseting BN statistics")
        model.cuda()
        update_bn_stats(model, trainloader)

    if args.export: # None
        torch.save(model.state_dict(), args.export + ".pth")
        return

    scales = (
        # [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        [1.0]
        if args.dataset == "citys"
        else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    )  
    
    evaluator = LSeg_MultiEvalModule(
        model, num_classes, scales=scales, flip=False
    ).cuda()
    evaluator.eval()
    
    # 5月7日修改内容，加上FPS计算功能
    total_time = 0.0  # To accumulate total processing time
    num_frames = 0  # To count the number of frames processed
    
    # 返回值为pixAcc, mIoU，前提是得有target输入。
    metric = utils.SegmentationMetric(testset.num_class)
    tbar = tqdm(test_data)

    f = open("logs/log_test_{}_{}.txt".format(args.jobname, args.dataset), "a+")
    per_class_iou = np.zeros(testset.num_class)
    cnt = 0


    for i, (image, gtmask, dst) in enumerate(tbar): # image [3, 512, 768], dst [512,768]
        start_time = datetime.datetime.now()
        print('test_lseg.py, #################### iteration ', i, ' ##########################')
        # print('test_lseg.py, iteration, image ', image[0].shape)
        # print('test_lseg.py, iteration, gtmask shape', gtmask[0].shape) # 有-1值，是0-1得到的，即背景，0值代表第一个类别
        # print('test_lseg.py, iteration, dst shape', dst)
        # print('args.eval ', args.eval)
        if args.eval:
            # 有label，可以作精度评价
            with torch.no_grad():
                if False:
                    sample = {"image": image[0].cpu().permute(1, 2, 0).numpy()}
                    out = torch.zeros(
                        1, testset.num_class, image[0].shape[1], image[0].shape[2]
                    ).cuda()

                    H, W = image[0].shape[1], image[0].shape[2]
                    for scale in scales:
                        long_size = int(math.ceil(520 * scale))
                        if H > W:
                            height = long_size
                            width = int(1.0 * W * long_size / H + 0.5)
                            short_size = width
                        else:
                            width = long_size
                            height = int(1.0 * H * long_size / W + 0.5)
                            short_size = height

                        rs = Resize(
                            width,
                            height,
                            resize_target=False,
                            keep_aspect_ratio=True,
                            ensure_multiple_of=32,
                            resize_method="minimal",
                            image_interpolation_method=cv2.INTER_AREA,
                        )

                        inf_image = (
                            torch.from_numpy(rs(sample)["image"])
                            .cuda()
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                        )
                        inf_image = torch.cat((inf_image, torch.fliplr(inf_image)), 0)
                        try:
                            pred = model(inf_image)
                        except:
                            print(H, W, sz, i)
                            exit()

                        pred0 = F.softmax(pred[0], dim=1)
                        pred1 = F.softmax(pred[1], dim=1)

                        pred = pred0 + 0.2 * pred1

                        out += F.interpolate(
                            pred.sum(0, keepdim=True),
                            (out.shape[2], out.shape[3]),
                            mode="bilinear",
                            align_corners=True,
                        )

                    predicts = [out]
                else:
                    # 并行forward，多GPU
                    # image传入 
                    predicts = evaluator.parallel_forward(image) # [1, 150, 512, 768]

                # metric.update(gtmask, predicts)
                # pixAcc, mIoU = metric.get()
                
                _, _, total_inter, total_union = metric.get_all()
                per_class_iou += 1.0 * total_inter / (np.spacing(1) + total_union)
                cnt+=1
                
                # tbar.set_description("pixAcc: %.4f, mIoU: %.4f" % (pixAcc, mIoU))

                # write out
                predict_out = [
                    testset.make_pred(torch.max(predict, 1)[1].cpu().numpy())
                    for predict in predicts
                ]
                # print('predict_out shape is , ', predict_out[0].shape)
                
                # 统计每一张图片的推理时间
                end_time = datetime.datetime.now()
                # print("Elapsed Time:", end_time - start_time) # 1除以这个时间，就是FPS

                outdir = "pred_output"
                # labelrgbdir = "labeldir_rgb"
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                
                # 输出结果
                for predict, impath in zip(predict_out, dst):
                    mask = utils.get_mask_pallete(predict, args.dataset)
                    outname = os.path.splitext(impath)[0] + ".png"
                    mask.save(os.path.join(outdir, outname))
                
                

        else:
            with torch.no_grad():
                outputs = evaluator.parallel_forward(image)
                predicts = [
                    testset.make_pred(torch.max(output, 1)[1].cpu().numpy())
                    for output in outputs
                ]
                print('test_lseg.py, test split, predicts shape ', predicts.shape)

            # output folder
            outdir = "outdir_ours"
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            for predict, impath in zip(predicts, dst):
                mask = utils.get_mask_pallete(predict, args.dataset)
                outname = os.path.splitext(impath)[0] + ".png"
                mask.save(os.path.join(outdir, outname))

    # if args.eval:
        # each_classes_iou = per_class_iou/cnt
        # print("pixAcc: %.4f, mIoU: %.4f" % (pixAcc, mIoU))
        # print('each_classes_iou ', each_classes_iou)
        # f.write("dataset {} ==> pixAcc: {:.4f}, mIoU: {:.4f}\n".format(args.dataset, pixAcc, mIoU))
        # for per_iou in each_classes_iou: f.write('{:.4f}, '.format(per_iou))
        # f.write('\n')


class ReturnFirstClosure(object):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        outputs = self._data[idx]
        return outputs[0]


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count() 
    test(args)
