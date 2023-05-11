import torch
import torch.nn as nn


# CBL -> Conv+BN+LeakyReLU
class CBL(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, negative_slope=0.1):
        super(CBL, self).__init__(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                                  nn.BatchNorm1d(out_channels),
                                  nn.LeakyReLU(negative_slope))


# 残差块，channels 包括输入通道和输出通道
class BasicBlock(nn.Module):
    def __init__(self, channels):
        super(BasicBlock, self).__init__()

        # block1 降通道数，block2 再将通道数升回去，如 64->32->64
        self.conv = nn.Sequential(CBL(channels[1], channels[0], kernel_size=1, padding=0),
                                  CBL(channels[0], channels[1]))

    def forward(self, x):
        return self.conv(x) + x


# channels: [in_channels, out_channels]; blocks: 指定每个 layer 中使用的残差块个数
def make_layer(channels, blocks):
    # 在每一个layer里面，首先利用一个步长为 2 的 3x3 卷积进行下采样
    layers = [CBL(channels[0], channels[1], stride=2)]
    for _ in range(0, blocks):
        layers.append(BasicBlock(channels))
    return nn.Sequential(*layers)


# darknet53
class DarkNet(nn.Module):
    def __init__(self, layers, num_classes=1):
        super(DarkNet, self).__init__()
        self.layer0 = CBL(8, 32)                         # (8, 96)   -> (32, 96)
        self.layer1 = make_layer([32, 64], layers[0])     # (32, 96)  -> (64, 48)
        self.layer2 = make_layer([64, 128], layers[1])    # (64, 48)  -> (128, 24)
        # out_1
        self.layer3 = make_layer([128, 256], layers[2])   # (128, 24) -> (256, 12)
        # out_2
        self.layer4 = make_layer([256, 512], layers[3])   # (256, 12)  -> (512, 6)
        # out_3
        self.layer5 = make_layer([512, 1024], layers[4])  # (512, 6)  -> (1024, 3)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(1024, num_classes),
                                        nn.Softmax(dim=1))

        self.initialize_weights()

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out_1 = self.layer3(x)
        out_2 = self.layer4(out_1)
        out_3 = self.layer5(out_2)
        out = self.avg_pool(out_3)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out_1, out_2, out_3, out

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
    def __init__(self, CBL_num=1):
        super(FPN, self).__init__()
        self.CBL_1 = CBL(1024, 256)
        self.CBL_2 = CBL(512, 256)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")  # [B, C, Length]
        out_channel = 256
        out_layers = [CBL(768, out_channel)] + [CBL(out_channel, out_channel) for _ in range(CBL_num - 1)]
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
    x_input = torch.randn(1, 8, 96)  # 创建随机输入 (batch_size, channels, width, height)
    darknet = DarkNet([1, 2, 8, 8, 4])     # [1, 2, 8, 8, 4] 分别指定残差块重复次数
    fpn = FPN(CBL_num=3)

    out1, out2, out3, out_ = darknet(x_input)
    print(f'out1: {out1.shape} \nout2: {out2.shape} \nout3: {out3.shape} \nout: {out_.shape}')
    out_fpn = fpn(out1, out2, out3)
    print(f'out1: {out_fpn.shape}')
