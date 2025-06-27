import torch
import torch.nn as nn
import torch.nn.functional as F

def up_sample(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    将特征图通过上采样恢复到目标尺寸。

    Args:
        src (Tensor): 原始图像。
        target (Tensor): 目标图像，用于计算大小。

    Returns:
        Tensor: 目标图像。
    """
    out = F.interpolate(src, target.shape[2:], mode="bilinear")

    return out


class REBNConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_rate=1):
        """
        Conv2d + BN + ReLU层。

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


class RSU7(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        """
        U2Net第一层中的7层Residual Block结构。

        Args:
            in_channels: 输入图像通道数。
            middle_channels: 网络中间层图像通道数。
            out_channels: 输出图像通道数。
        """
        super(RSU7, self).__init__()

        self.layer_in = REBNConv(in_channels, out_channels)

        # 向下部分
        self.input_layer_encoder = REBNConv(out_channels, middle_channels)
        self.middle_layer_encoder = REBNConv(middle_channels, middle_channels)
        self.out_layer_encoder = REBNConv(middle_channels, middle_channels, dilation_rate=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 向上取整

        # 向上部分
        self.middle_layer_decoder = REBNConv(middle_channels * 2, middle_channels)
        self.out_layer_decoder = REBNConv(middle_channels * 2, out_channels)

    def forward(self, x):
        x_in = self.layer_in(x)

        # ----- Encoder -----
        # Layer 1
        x1_encoder = self.input_layer_encoder(x_in)
        hx = self.pool(x1_encoder)

        # Layer 2
        x2_encoder = self.middle_layer_encoder(hx)
        hx = self.pool(x2_encoder)

        # Layer 3
        x3_encoder = self.middle_layer_encoder(hx)
        hx = self.pool(x3_encoder)

        # Layer 4
        x4_encoder = self.middle_layer_encoder(hx)
        hx = self.pool(x4_encoder)

        # Layer 5
        x5_encoder = self.middle_layer_encoder(hx)
        hx = self.pool(x5_encoder)

        # Layer 6
        x6_encoder = self.middle_layer_encoder(hx)

        # Layer 7
        x7_encoder = self.out_layer_encoder(x6_encoder)

        # ----- Decoder -----
        # Layer 6
        hx = self.middle_layer_decoder(torch.cat((x7_encoder, x6_encoder), 1))  # 通道拼接
        hx = up_sample(hx, x5_encoder)

        # Layer 5
        hx = self.middle_layer_decoder(torch.cat((hx, x5_encoder), 1))
        hx = up_sample(hx, x4_encoder)

        # Layer 4
        hx = self.middle_layer_decoder(torch.cat((hx, x4_encoder), 1))
        hx = up_sample(hx, x3_encoder)

        # Layer 3
        hx = self.middle_layer_decoder(torch.cat((hx, x3_encoder), 1))
        hx = up_sample(hx, x2_encoder)

        # Layer 2
        hx = self.middle_layer_decoder(torch.cat((hx, x2_encoder), 1))
        hx = up_sample(hx, x1_encoder)

        # Layer 1
        hx = self.out_layer_decoder(torch.cat((hx, x1_encoder), 1))
        hx = hx + x_in

        return hx


class RSU6(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        """
        U2Net第二层中的6层Residual Block结构。

        Args:
            in_channels: 输入图像通道数。
            middle_channels: 网络中间层图像通道数。
            out_channels: 输出图像通道数。
        """
        super(RSU6, self).__init__()

        self.layer_in = REBNConv(in_channels, out_channels)

        # 向下部分
        self.input_layer_encoder = REBNConv(out_channels, middle_channels)
        self.middle_layer_encoder = REBNConv(middle_channels, middle_channels)
        self.out_layer_encoder = REBNConv(middle_channels, middle_channels, dilation_rate=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 向上取整

        # 向上部分
        self.middle_layer_decoder = REBNConv(middle_channels * 2, middle_channels)
        self.out_layer_decoder = REBNConv(middle_channels * 2, out_channels)

    def forward(self, x):
        x_in = self.layer_in(x)

        # ----- Encoder -----
        # Layer 1
        x1_encoder = self.input_layer_encoder(x_in)
        hx = self.pool(x1_encoder)  # type: ignore

        # Layer 2
        x2_encoder = self.middle_layer_encoder(hx)
        hx = self.pool(x2_encoder)  # type: ignore

        # Layer 3
        x3_encoder = self.middle_layer_encoder(hx)
        hx = self.pool(x3_encoder)  # type: ignore

        # Layer 4
        x4_encoder = self.middle_layer_encoder(hx)
        hx = self.pool(x4_encoder)  # type: ignore

        # Layer 5
        x5_encoder = self.middle_layer_encoder(hx)

        # Layer 6
        x6_encoder = self.out_layer_encoder(x5_encoder)

        # ----- Decoder -----
        # Layer 5
        hx = self.middle_layer_decoder(torch.cat((x6_encoder, x5_encoder), 1))
        hx = up_sample(hx, x4_encoder)

        # Layer 4
        hx = self.middle_layer_decoder(torch.cat((hx, x4_encoder), 1))
        hx = up_sample(hx, x3_encoder)

        # Layer 3
        hx = self.middle_layer_decoder(torch.cat((hx, x3_encoder), 1))
        hx = up_sample(hx, x2_encoder)

        # Layer 2
        hx = self.middle_layer_decoder(torch.cat((hx, x2_encoder), 1))
        hx = up_sample(hx, x1_encoder)

        # Layer 1
        hx = self.out_layer_decoder(torch.cat((hx, x1_encoder), 1))
        hx = hx + x_in

        return hx


class RSU5(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        """
        U2Net第三层中的5层Residual Block结构。

        Args:
            in_channels: 输入图像通道数。
            middle_channels: 网络中间层图像通道数。
            out_channels: 输出图像通道数。
        """
        super(RSU5, self).__init__()

        self.layer_in = REBNConv(in_channels, out_channels)

        # 向下部分
        self.input_layer_encoder = REBNConv(out_channels, middle_channels)
        self.middle_layer_encoder = REBNConv(middle_channels, middle_channels)
        self.out_layer_encoder = REBNConv(middle_channels, middle_channels, dilation_rate=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 向上取整

        # 向上部分
        self.middle_layer_decoder = REBNConv(middle_channels * 2, middle_channels)
        self.out_layer_decoder = REBNConv(middle_channels * 2, out_channels)

    def forward(self, x):
        x_in = self.layer_in(x)

        # ----- Encoder -----
        # Layer 1
        x1_encoder = self.input_layer_encoder(x_in)
        hx = self.pool(x1_encoder)

        # Layer 2
        x2_encoder = self.middle_layer_encoder(hx)
        hx = self.pool(x2_encoder)

        # Layer 3
        x3_encoder = self.middle_layer_encoder(hx)
        hx = self.pool(x3_encoder)

        # Layer 4
        x4_encoder = self.middle_layer_encoder(hx)

        # Layer 5
        x5_encoder = self.out_layer_encoder(x4_encoder)

        # ----- Decoder -----
        # Layer 4
        hx = self.middle_layer_decoder(torch.cat((x5_encoder, x4_encoder), 1))
        hx = up_sample(hx, x3_encoder)

        # Layer 3
        hx = self.middle_layer_decoder(torch.cat((hx, x3_encoder), 1))
        hx = up_sample(hx, x2_encoder)

        # Layer 2
        hx = self.middle_layer_decoder(torch.cat((hx, x2_encoder), 1))
        hx = up_sample(hx, x1_encoder)

        # Layer 1
        hx = self.out_layer_decoder(torch.cat((hx, x1_encoder), 1))
        hx = hx + x_in

        return hx


class RSU4(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        """
        U2Net第四层中的4层Residual Block结构。

        Args:
            in_channels: 输入图像通道数。
            middle_channels: 网络中间层图像通道数。
            out_channels: 输出图像通道数。
        """
        super(RSU4, self).__init__()

        self.layer_in = REBNConv(in_channels, out_channels)

        # 向下部分
        self.input_layer_encoder = REBNConv(out_channels, middle_channels)
        self.middle_layer_encoder = REBNConv(middle_channels, middle_channels)
        self.out_layer_encoder = REBNConv(middle_channels, middle_channels, dilation_rate=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 向上取整

        # 向上部分
        self.middle_layer_decoder = REBNConv(middle_channels * 2, middle_channels)
        self.out_layer_decoder = REBNConv(middle_channels * 2, out_channels)

    def forward(self, x):
        x_in = self.layer_in(x)

        # ----- Encoder -----
        # Layer 1
        x1_encoder = self.input_layer_encoder(x_in)
        hx = self.pool(x1_encoder)

        # Layer 2
        x2_encoder = self.middle_layer_encoder(hx)
        hx = self.pool(x2_encoder)

        # Layer 3
        x3_encoder = self.middle_layer_encoder(hx)

        # Layer 4
        x4_encoder = self.out_layer_encoder(hx)

        # ----- Decoder -----
        # Layer 3
        hx = self.middle_layer_decoder(torch.cat((x4_encoder, x3_encoder), 1))
        hx = up_sample(hx, x2_encoder)

        # Layer 2
        hx = self.middle_layer_decoder(torch.cat((hx, x2_encoder), 1))
        hx = up_sample(hx, x1_encoder)

        # Layer 1
        hx = self.out_layer_decoder(torch.cat((hx, x1_encoder), 1))
        hx = hx + x_in

        return hx


class RSU4F(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        """
        U2Net第五层中的特殊4层Residual Block结构。

        Args:
            in_channels: 输入图像通道数。
            middle_channels: 网络中间层图像通道数。
            out_channels: 输出图像通道数。
        """
        super(RSU4F, self).__init__()

        self.layer_in = REBNConv(in_channels, out_channels)

        # 向下部分
        self.layer1_encoder = REBNConv(out_channels, middle_channels)
        self.layer2_encoder = REBNConv(middle_channels, middle_channels, dilation_rate=2)
        self.layer3_encoder = REBNConv(middle_channels, middle_channels, dilation_rate=4)
        self.layer4_encoder = REBNConv(middle_channels, middle_channels, dilation_rate=8)

        # 向上部分
        self.layer3_decoder = REBNConv(middle_channels * 2, middle_channels, dilation_rate=4)
        self.layer2_decoder = REBNConv(middle_channels * 2, middle_channels, dilation_rate=2)
        self.layer1_decoder = REBNConv(middle_channels * 2, out_channels)

    def forward(self, x):
        x_in = self.layer_in(x)

        # ----- Encoder -----
        # Layer 1
        x1_encoder = self.layer1_encoder(x_in)

        # Layer 2
        x2_encoder = self.layer2_encoder(x1_encoder)

        # Layer 3
        x3_encoder = self.layer3_encoder(x2_encoder)

        # Layer 4
        x4_encoder = self.layer4_encoder(x3_encoder)

        # ----- Decoder -----
        # Layer 3
        hx = self.layer3_decoder(torch.cat((x4_encoder, x3_encoder), 1))

        # Layer 2
        hx = self.layer2_decoder(torch.cat((hx, x2_encoder), 1))

        # Layer 1
        hx = self.layer1_decoder(torch.cat((hx, x1_encoder), 1))
        hx = hx + x_in

        return hx


class U2NET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        """
        完整版的U2Net，输入为3通道彩色图像，输出为单通道分割掩码。

        Args:
            in_channels: 输入图像通道数。
            out_channels: 输出图像通道数。
        """
        super(U2NET, self).__init__()

        # 向下部分
        self.stage1_encoder = RSU7(in_channels, 32, 64)
        self.stage2_encoder = RSU6(64, 32, 128)
        self.stage3_encoder = RSU5(128, 64, 256)
        self.stage4_encoder = RSU4(256, 128, 512)
        self.stage5_encoder = RSU4F(512, 256, 512)
        self.stage6_encoder = RSU4F(512, 256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # 向上部分
        self.stage5_decoder = RSU4F(1024, 256, 512)
        self.stage4_decoder = RSU4(1024, 128, 256)
        self.stage3_decoder = RSU5(512, 64, 128)
        self.stage2_decoder = RSU6(256, 16, 64)
        self.stage1_decoder = RSU5(128, 16, 64)

        # 每层特征图
        self.side1 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.side2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.side3 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        self.side4 = nn.Conv2d(256, out_channels, kernel_size=3, padding=1)
        self.side5 = nn.Conv2d(512, out_channels, kernel_size=3, padding=1)
        self.side6 = nn.Conv2d(512, out_channels, kernel_size=3, padding=1)

        # 输出
        self.output = nn.Conv2d(out_channels * 6, out_channels, kernel_size=1)

    def forward(self, x):
        # ----- Encoder -----
        # stage 1
        x1_encoder = self.stage1_encoder(x)
        hx = self.pool(x1_encoder)

        # stage 2
        x2_encoder = self.stage2_encoder(hx)
        hx = self.pool(x2_encoder)

        # stage 3
        x3_encoder = self.stage3_encoder(hx)
        hx = self.pool(x3_encoder)

        # stage 4
        x4_encoder = self.stage4_encoder(hx)
        hx = self.pool(x4_encoder)

        # stage 5
        x5_encoder = self.stage5_encoder(hx)
        hx = self.pool(x5_encoder)

        # stage 6
        x6_encoder = self.stage6_encoder(hx)
        hx = up_sample(x6_encoder, x5_encoder)

        # ----- Decoder -----
        # stage 5
        x5_decoder = self.stage5_decoder(torch.cat((hx, x5_encoder), 1))
        hx = up_sample(x5_decoder, x4_encoder)

        # stage 4
        x4_decoder = self.stage4_decoder(torch.cat((hx, x4_encoder), 1))
        hx = up_sample(x4_decoder, x3_encoder)

        # stage 3
        x3_decoder = self.stage3_decoder(torch.cat((hx, x3_encoder), 1))
        hx = up_sample(x3_decoder, x2_encoder)

        # stage 2
        x2_decoder = self.stage2_decoder(torch.cat((hx, x2_encoder), 1))
        hx = up_sample(x2_decoder, x1_encoder)

        # stage 1
        x1_decoder = self.stage1_decoder(torch.cat((hx, x1_encoder), 1))

        # ----- Side Output -----
        # stage 1
        sup1 = self.side1(x1_decoder)

        # stage 2
        sup2 = self.side2(x2_decoder)
        sup2 = up_sample(sup2, sup1)

        # stage 3
        sup3 = self.side3(x3_decoder)
        sup3 = up_sample(sup3, sup1)

        # stage 4
        sup4 = self.side4(x4_decoder)
        sup4 = up_sample(sup4, sup1)

        # stage 5
        sup5 = self.side5(x5_decoder)
        sup5 = up_sample(sup5, sup1)

        # stage 6
        sup6 = self.side6(x6_encoder)
        sup6 = up_sample(sup6, sup1)

        # Output
        sup0 = self.output(torch.cat((sup1, sup2, sup3, sup4, sup5, sup6), 1))

        return F.sigmoid(sup0), F.sigmoid(sup1), F.sigmoid(sup2), F.sigmoid(sup3), F.sigmoid(sup4), F.sigmoid(sup5), F.sigmoid(sup6)