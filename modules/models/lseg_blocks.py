import torch
import torch.nn as nn

from .lseg_vit import (
    _make_pretrained_clip_vitl16_384,
    _make_pretrained_clip_vitb32_384,
    _make_pretrained_clipRN50x16_vitl16_384,
    forward_vit,
)

# 创建一个encoder
# 包含clip预训练模型，vit预训练模型，和4层卷积模型
def _make_encoder(
    backbone,
    features, # 256
    use_pretrained=True,
    groups=1,
    expand=False,
    exportable=True,
    hooks=None,
    use_vit_only=False,
    use_readout="ignore",
    enable_attention_hooks=False,
):  
    if backbone == "clip_vitl16_384": 
        clip_pretrained, pretrained = _make_pretrained_clip_vitl16_384(
            use_pretrained, # 使用预训练模型
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        # 虽然已经有一个预训练的 ViT 模型，但这个函数 _make_scratch 通过定义四个卷积层，
        # 可以将输入特征映射到新的特征空间中，并且这些特征的通道数可以根据需要进行扩展。
        # 这种映射和扩展可以在特定的下游任务中提供更加丰富和多样化的特征表示，有助于提升模型的性能。
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        ) 

    elif backbone == "clipRN50x16_vitl16_384":
        clip_pretrained, pretrained = _make_pretrained_clipRN50x16_vitl16_384(
            use_pretrained,
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        )
    elif backbone == "clip_vitb32_384":
        clip_pretrained, pretrained = _make_pretrained_clip_vitb32_384(
            use_pretrained, 
            hooks=hooks, 
            use_readout=use_readout,
        )
        scratch = _make_scratch(
            [96, 192, 384, 768], features, groups=groups, expand=expand
        ) 
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False

    return clip_pretrained, pretrained, scratch



# 定义了一个自定义的神经网络模块，该模块包含多个卷积层
def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    # 定义四个卷积层
    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    return scratch


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x

# 是一个自定义的神经网络模块，用于融合不同层次的特征，就是将xs[0]和xs[1]融合了
class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features, # 特征通道数
        activation, 
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        
        # 定义模块
        # 将特征映射到输出特征空间, 即last layer
        self.out_conv = nn.Conv2d(
            features, # 输入通道
            out_features, # 输出通道
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)# 输入通道，激活，批归一化
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    # 拼接模块
    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        # 观察这个输入是哪里来的？是否有输入传给FeatureFusionBlock_custom
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output

