import torch
from torch import nn
from torch.nn import functional as func


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        input: (batch, feature, sequence)
        output: (batch, feature, sequence)
        :param input_size: (batch size, feature size, sequence length)
        :param hidden_size: (hidden_feature, num_layers)
        :param output_size: (feature size, sequence length)
        """
        super().__init__()
        # store
        self.num_directions = 1  # 单向LSTM
        self.hidden_feature, self.num_layers = hidden_size
        self.batch_size, self.feature_size, self.seq_len = input_size
        # build
        self.lstm = nn.LSTM(self.feature_size, self.hidden_feature, self.num_layers, batch_first=True)
        self.linear_1 = nn.Linear(self.hidden_feature, output_size[0])
        self.linear_2 = nn.Linear(self.seq_len, output_size[1])
        # init
        self.h_0 = None
        self.c_0 = None
        self.rebuild_hc()

    def rebuild_hc(self, device='cpu'):
        self.h_0 = torch.ones(self.num_directions * self.num_layers,
                              self.batch_size, self.hidden_feature).to(device)
        self.c_0 = torch.ones(self.num_directions * self.num_layers,
                              self.batch_size, self.hidden_feature).to(device)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)  # (batch, seq_len, feature)
        output, last_state = self.lstm(x, (self.h_0, self.c_0))
        pred = func.silu(self.linear_1(output))  # (batch, seq_len, feature)
        pred = torch.transpose(pred, 2, 1)  # (batch, feature, seq_len)
        pred = self.linear_2(pred)
        return pred


if __name__ == '__main__':
    input_x = torch.tensor([[[1, 2, 4, 8, 16]],
                            [[2, 4, 8, 16, 32]]], dtype=torch.float32)
    net = LSTM(input_size=(2, 1, 5), hidden_size=(4, 6), output_size=(1, 1))

    with torch.no_grad():
        res = net(input_x)
    print(res)
