from torch import nn

from CLNet.darknet import DarkNet, FPN
from CLNet.lstm import LSTM


class CLNet(nn.Module):
    def __init__(self, params):
        super(CLNet, self).__init__()
        self.darknet = DarkNet(params)  # [1, 2, 8, 8, 4] 分别指定残差块重复次数
        self.fpn = FPN(params)
        self.lstm = LSTM(params)

    def forward(self, x):
        x_1, x_2, x_3 = self.darknet(x)
        x = self.fpn(x_1, x_2, x_3)
        x = self.lstm(x)
        return x


if __name__ == '__main__':
    import torch

    class Test_Options:
        def __init__(self):
            # darknet
            self.in_channel = 6
            self.base_channel = 36
            self.layers = [1, 2, 8, 8, 4]  # 分别指定残差块重复次数
            # fpn
            self.avg_channel = 288
            self.CBL_num = 3
            # lstm
            self.input_size = (1, 72, 12)
            self.hidden_size = (72, 3)
            self.output_size = (1, 9)

    test_options = Test_Options()
    x_test = torch.ones((1, 6, 96), dtype=torch.float32)
    ANN = CLNet(test_options)
    with torch.no_grad():
        y_test = ANN(x_test)
    print(y_test)
