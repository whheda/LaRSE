# 📦 Visual-language reasoning segmentation of function-level building footprint (LaRSE)

Function-level building footprint is vital for understanding socio-economic heterogeneity and advancing sustainable development. However, typical researches are restricted to either parcel-level urban function identification or binary building footprint extraction, neither of which can delineate human activity spaces at the footprint unit. Besides, common segmentation networks lack human-like reasoning capabilities, as they fail to incorporate surrounding entities or leverage geo-spatial knowledge. 

To this being, this study proposes a function-level building footprint extraction based on visual-language reasoning segmentation framework (LaRSE). It simplifies the task into two stages, i.e., visual model for segmenting building footprint and deriving context embeddings, followed by a language model that utilizes these embeddings to perform precise semantic reasoning for each footprint. In this framework, the visual model acts as the "eye" by perceiving input, while the language model functions as the "brain", interpreting these inputs logically.

![图1reasoning segmentation](https://github.com/user-attachments/assets/5f8987ed-a311-43fd-84e5-1aead39b7e1b)

## 🚀 Installation

```bash
conda install ipython
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/
pip install pytorch-lightning==1.3.5
pip install opencv-python
pip install imageio
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install altair
pip install streamlit
pip install --upgrade protobuf
pip install timm
pip install tensorboardX
pip install matplotlib
pip install test-tube
pip install wandb
```
specific process can be find in https://github.com/isl-org/lang-seg

## 🚀 Dataset Preparation

### 1. BUFF dataset download

In this study, we developed a solid benchmark dataset for deep learning-based building-footprint-scale function identification, named the Building Footprint Function (BuFF) dataset. After assigning function labels to each building polygon, we converted the building polygon in Shenzhen to raster format, and paired it with 0.5-meter resolution high-resolution Google Earth remote sensing images to construct the Building Footprint Function (BuFF) dataset. we cropped a total of 12,940 images at 512×512 pixels with 0.5-meter resolution, covering 500,000 buildings. The number of buildings for each function type is shown in the table. The dataset was split into a 7:1:2 ratio, with 9,056 images for training, 1,304 for validation, and 2,580 for testing. The BUFF dataset can be downloaded in https://doi.org/10.5281/zenodo.15433646.

![buff_dataset](https://github.com/user-attachments/assets/50b93e45-c5c0-443f-bf39-f45b7263dda1)

### 2. BUFF dataset loading

The user need to define the dataset class before loading the dataset. For example, the BUFF dataset is defined as the "buff1w" class in ```./data/__init__.py```. At the same time, place the ```buff1w.py``` file into the ```encoding``` library within the user-created venv, typically under the path ```./lib/python3.7/site-packages/encoding/datasets/```. Also, add the following line to the ```__init__.py``` file in that directory.

```bash
from .buff1w import BuFF1WChallengeDataset
datasets = {
    'coco': COCOSegmentation,
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'pcontext': ContextSegmentation,
    'citys': CitySegmentation,
    'imagenet': ImageNetDataset,
    'minc': MINCDataset,
    'cifar10': CIFAR10,
    'buff6k': BuFF6KChallengeDataset,
    'buff4k': BuFF4KChallengeDataset,
    'buff1w': BuFF1WChallengeDataset,
}
```

## 🚀 Pretrained model preparation

Loading pre-trained visual-language model RemoteCLIP. The path is ```./modules/models/lseg_vit.py```, and the default model used is ```RemoteCLIP-ViT-B-32.pt```, which can be replaced as needed. The download link for ```RemoteCLIP-ViT-B-32.pt``` can be found at: https://github.com/ChenDelong1999/RemoteCLIP. The download link for ```checkpoint_LARSE.ckpt``` model can be found at https://pan.baidu.com/s/1qCUb-4E7uyu0fo0M5GPjGQ?pwd=2su3, code: 2su3 


## 🚀 Test demo

Define the root path of the dataset and the path of the pre-trained weight ```checkpoint_LARSE.ckpt``` in ```test.sh```, and define the output path in ```test.py```. Then you can easily run the demo:

```bash
bash test.sh
```

If you find this repo useful, please cite:

```bash
author={Da He, Xiaoping Liu, Qian Shi, Yue Zheng}
title={Visual-language reasoning segmentation (LARSE) of function-level building footprint across Yangtze River Economic Belt of China},
journaltitle={Sustainable Cities and Society},
volume={127},
issue={106439},
year={2025},
ISSN={2210-6707},
DOI={https://doi.org/10.1016/j.scs.2025.106439}
```

# Acknowledge
Thanks to the code base from [Lseg](https://github.com/isl-org/lang-seg), [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding), [CLIP](https://github.com/openai/CLIP), [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP)
