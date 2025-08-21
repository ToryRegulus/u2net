import math
import re

import torch
from torch import nn, Tensor


def _upsample(x: Tensor, size: tuple[int, ...]) -> Tensor:
    """
    对当前图像上采样为指定大小。

    Args:
        x (Tensor): 被上采样的图像。
        size (tuple[int]): 上采样的目标大小。

    Returns:
        Tensor: 上采样后的图片。
    """
    return nn.Upsample(size=size, mode="bilinear")(x)


def _size_map(x: Tensor, height: int) -> dict[int, tuple[int, ...]]:
    """
    RSU每层输出图像的大小映射。

    Args:
        x (Tensor): 输入图像。
        height (int): 当前RSU模块的高度。

    Returns:
        dict[int, tuple]: 每层图像大小的映射字典。
    """
    if height <= 1:
        raise ValueError(f"RSU高度必须大于1，当前为 {height}")

    size_map: dict[int, tuple[int, ...]] = {}
    size = list(x.shape[-2:])

    for h in range(1, height):
        size_map[h] = tuple(size)
        size = [(i + 1) // 2 for i in size]

    return size_map


class REBNConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_rate=1):
        """
        初始化Conv2d + BN + ReLU层。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            kernel_size: 卷积核大小。
            dilation_rate: 膨胀系数。
        """
        super(REBNConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=dilation_rate, dilation=dilation_rate)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)  # 节省显存

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class RSU(nn.Module):
    def __init__(self, height: int, in_channels: int, middle_channels: int, out_channels: int, dilated: bool = False):
        """
        初始化每层U2Net中的RSU模块，不同层数的RSU模块内部的高度均有差别。

        Args:
            height (int): RSU模块的高度。
            in_channels (int): 输入图像通道数。
            middle_channels (int): 中间层图像通道数。
            out_channels (int): 输出图像通道数。
            dilated (bool): 是否进行膨胀卷积。
        """
        super(RSU, self).__init__()

        self.height = height
        self.dilated = dilated
        self.in_channels = in_channels
        self._make_layers(height, in_channels, middle_channels, out_channels)

    def forward(self, x):
        try:
            size_map = _size_map(x, self.height)
            x = self.layer_in(x)  # type: ignore

            def encoder_decoder(x: Tensor, height: int = 1) -> Tensor:
                """
                递归实现RSU模块内的编码器-解码器架构，不同高度的RSU模块有不同的内部结构，RSU-4F不需要池化与上采样。

                Args:
                    x (Tensor): 输入图像。
                    height (int): RSU模块的高度。

                Returns:
                    Tensor: 经过编码器-解码器处理后的图像输出。
                """
                if height < self.height:
                    x_encoder = getattr(self, f"layer_{height}")(x)
                    if not self.dilated and height < self.height - 1:
                        hx = encoder_decoder(getattr(self, f"downsample")(x_encoder), height + 1)
                    else:
                        hx = encoder_decoder(x_encoder, height + 1)

                    hx = getattr(self, f"layer_{height}d")(torch.cat((hx, x_encoder), dim=1))

                    if not self.dilated and height > 1:
                        return _upsample(hx, size_map[height - 1])
                    else:
                        return hx

                else:
                    return getattr(self, f"layer_{height}")(x)

            return x + encoder_decoder(x)

        except ValueError as e:
            print(f"前向推导数值异常: {e}")

            return None

    def _make_layers(self, height: int, in_channels: int, middle_channels: int, out_channels: int,
                     dilated: bool = False) -> None:
        """
        注册RUS模组的各个层，RSU-4F模块需要膨胀卷积。

        Args:
            height (int): RSU模块高度。
            in_channels (int): 输入图像通道数。
            middle_channels (int): 中间层图像通道数。
            out_channels (int): 输出图像通道数。
            dilated (bool): 是否进行膨胀卷积。
        """
        self.add_module("layer_in", REBNConv(in_channels, out_channels))
        self.add_module("downsample", nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))  # 向上取整
        self.add_module("layer_1", REBNConv(out_channels, middle_channels))
        self.add_module("layer_1d", REBNConv(middle_channels * 2, out_channels))

        for i in range(2, height):
            dilation_rate = 1 if not dilated else 2 ** (i - 1)
            self.add_module(f"layer_{i}", REBNConv(middle_channels, middle_channels, dilation_rate=dilation_rate))
            self.add_module(f"layer_{i}d", REBNConv(middle_channels * 2, middle_channels, dilation_rate=dilation_rate))

        dilation_rate = 2 if not dilated else 2 ** (height - 1)
        self.add_module(f"layer_{height}", REBNConv(middle_channels, middle_channels, dilation_rate=dilation_rate))


class U2NET(nn.Module):
    def __init__(self, cfg: dict[str, list]):
        """
        U2Net各层结构的初始化。

        Args:
            cfg (dict): U2Net每层结构的参数。
        """
        super(U2NET, self).__init__()

        self.height: int = 0  # U2Net总高度
        self._make_layers(cfg)

    def forward(self, x):
        s_up = []  # 每层特征图的暂存列表
        size_map = _size_map(x, self.height)

        def encoder_decoder(x: Tensor, height: int = 1) -> Tensor:
            """
            递归实现U2Net整体的编码器-解码器架构，同时记录每层输出的特征图，存入暂存列表中。

            Args:
                x (Tensor): 输入图像。
                height (int): 编解码器高度。

            Returns:
                Tensor: 每一层解码器输出的图像。
            """
            if height < 6:
                x_encoder = getattr(self, f"stage{height}")(x)
                hx = encoder_decoder(getattr(self, f"downsample")(x_encoder), height + 1)
                x_decoder = getattr(self, f"stage{height}d")(torch.cat((hx, x_encoder), dim=1))
                side(x_decoder, height)

                if height > 1:
                    return _upsample(x_decoder, size_map[height - 1])
                else:
                    return x_decoder

            else:
                x_encoder = getattr(self, f"stage{height}")(x)
                side(x_encoder, height)

                return _upsample(x_encoder, size_map[height - 1])

        def side(x, height: int) -> None:
            """
            提取解码器侧，每一层的特征图，将特征图存入暂存列表中以便后续拼接激活。

            Args:
                x: 输入图像。
                height (int): 输出图像。
            """
            hx = getattr(self, f"side_{height}")(x)
            hx = _upsample(hx, size_map[1])
            s_up.append(hx)

        def fuse() -> list[Tensor]:
            """
            融合各层特征图并激活输出。

            Returns:
                list: 混合输出与各层特征图激活后的列表。
            """
            s_up.reverse()
            s_fuse = getattr(self, "fuse")(torch.cat(s_up, dim=1))
            s_up.insert(0, s_fuse)

            return s_up

        encoder_decoder(x)

        return fuse()

    def _make_layers(self, cfg: dict[str, list]) -> None:
        """
        注册U2Net每层的RSU模块及每层图像融合模块。

        Args:
            cfg (dict): U2Net每层结构的参数。
        """
        self.height = (len(cfg) + 1) // 2

        self.add_module("downsample", nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        for k, v in cfg.items():
            self.add_module(k, RSU(*v[0]))

            if v[1] != -1:
                match = re.search(r'\d', k)
                if match:
                    self.add_module(f"side_{match.group()}", nn.Conv2d(v[1], 1, kernel_size=3, padding=1))
                else:
                    self.add_module(f"side_{k}", nn.Conv2d(v[1], 1, kernel_size=3, padding=1))

        self.add_module("fuse", nn.Conv2d(self.height, 1, kernel_size=1))
