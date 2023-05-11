import torch
import torch.nn as nn


# CBL -> Conv+BN+LeakyReLU
class CBL(nn.Sequential):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, bias=False, negative_slope=0.1):
        super(CBL, self).__init__(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                                  nn.BatchNorm1d(out_channels),
                                  nn.LeakyReLU(negative_slope))


# 将对每个通道进行深度可分离卷积
class CBL_ds(nn.Sequential):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, bias=False, negative_slope=0.1):
        super(CBL_ds, self).__init__(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding,
                                               bias=bias, groups=in_channels),
                                     nn.BatchNorm1d(out_channels),
                                     nn.LeakyReLU(negative_slope))


# 残差块，channels 包括输入通道和输出通道
class BasicBlock(nn.Module):
    def __init__(self, channels, depth_separable):
        super(BasicBlock, self).__init__()

        # block1 降通道数，block2 再将通道数升回去，如 64->32->64
        if depth_separable:
            self.conv = nn.Sequential(CBL_ds(channels[1], channels[0], kernel_size=1, padding=0),
                                      CBL(channels[0], channels[1]))
        else:
            self.conv = nn.Sequential(CBL(channels[1], channels[0], kernel_size=1, padding=0),
                                      CBL(channels[0], channels[1]))

    def forward(self, x):
        return self.conv(x) + x


# channels: [in_channels, out_channels]; blocks: 指定每个 layer 中使用的残差块个数
def make_layer(channels, blocks, depth_separable=False):
    # 在每一个layer里面，首先利用一个步长为 2 的 3x3 卷积进行下采样
    layers = [CBL(channels[0], channels[1], stride=2)]
    for _ in range(0, blocks):
        layers.append(BasicBlock(channels, depth_separable))
    return nn.Sequential(*layers)


class SPP(nn.Module):
    def __init__(self, out_channels):
        super(SPP, self).__init__()
        self.pool_1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)  # padding=3 // 2
        self.pool_2 = nn.MaxPool1d(kernel_size=5, stride=1, padding=2)  # padding=5 // 2
        self.pool_3 = nn.MaxPool1d(kernel_size=9, stride=1, padding=4)  # padding=9 // 2
        self.layer_out = CBL(4 * out_channels, out_channels)

    def forward(self, x):
        x_1 = self.pool_1(x)
        x_2 = self.pool_2(x)
        x_3 = self.pool_3(x)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)
        x = self.layer_out(x)
        return x


# darknet53
class DarkNet(nn.Module):
    def __init__(self, params):
        super(DarkNet, self).__init__()
        self.layer_0 = CBL_ds(params.in_channel, params.base_channel)
        self.layer_1 = make_layer([params.base_channel, params.base_channel * 2], params.layers[0],
                                  depth_separable=False)
        self.layer_2 = make_layer([params.base_channel * 2, params.base_channel * 4], params.layers[1],
                                  depth_separable=False)
        # SPP
        self.spp = SPP(params.base_channel * 4)
        # out_1
        self.layer_3 = make_layer([params.base_channel * 4, params.base_channel * 8], params.layers[2],
                                  depth_separable=False)
        # out_2
        self.layer_4 = make_layer([params.base_channel * 8, params.base_channel * 16], params.layers[3],
                                  depth_separable=False)
        # out_3
        self.layer_5 = make_layer([params.base_channel * 16, params.base_channel * 32], params.layers[4],
                                  depth_separable=False)
        self.initialize_weights()

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.spp(x)
        out_1 = self.layer_3(x)
        out_2 = self.layer_4(out_1)
        out_3 = self.layer_5(out_2)
        return out_1, out_2, out_3

    # 权值初始化
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class FPN(nn.Module):
    def __init__(self, params):
        super(FPN, self).__init__()
        avg_channel = params.avg_channel  # out_1
        self.CBL_1 = CBL(avg_channel * 4, avg_channel)  # out_3
        self.CBL_2 = CBL(avg_channel * 2, avg_channel)  # out_2
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")  # [B, C, Length]
        out_layers = [CBL(3 * avg_channel, avg_channel)] + [CBL(avg_channel // 2 ** i, avg_channel // 2 ** (i + 1))
                                                            for i in range(params.CBL_num-1)]
        self.CBL_out = nn.Sequential(*out_layers)

    def forward(self, x_1, x_2, x_3):
        x_3 = self.CBL_1(x_3)
        x_3 = self.upsample(self.upsample(x_3))
        x_2 = self.CBL_2(x_2)
        x_2 = self.upsample(x_2)
        # scale fusion
        x_2 = x_2 + x_3
        x_1 = x_1 + x_2
        # scale mosaic
        x = torch.cat((x_1, x_2, x_3), dim=1)
        x = self.CBL_out(x)
        return x


if __name__ == '__main__':
    pass
