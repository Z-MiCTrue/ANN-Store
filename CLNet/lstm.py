import torch
from torch import nn
from torch.nn import functional as func


class LSTM(nn.Module):
    def __init__(self, params):
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
        self.hidden_feature, self.num_layers = params.hidden_size
        self.batch_size, self.feature_size, self.seq_len = params.input_size
        # build
        self.lstm = nn.LSTM(self.feature_size, self.hidden_feature, self.num_layers, batch_first=True)
        self.linear_1 = nn.Linear(self.hidden_feature, params.output_size[0])
        self.linear_1_ = nn.Linear(params.output_size[0], params.output_size[0])
        self.linear_2 = nn.Linear(self.seq_len, params.output_size[1])
        self.linear_2_ = nn.Linear(params.output_size[1], params.output_size[1])

    def forward(self, x):
        x = torch.transpose(x, 1, 2)  # (batch, seq_len, feature)
        output, last_state = self.lstm(x)
        # 1
        pred = func.silu(self.linear_1(output))  # (batch, seq_len, feature)
        pred_ = func.silu(self.linear_1_(pred))
        pred = pred + pred_
        pred = torch.transpose(pred, 2, 1)  # (batch, feature, seq_len)
        # 2
        pred = self.linear_2(pred)
        pred_ = func.silu(self.linear_2_(pred))
        pred = pred + pred_
        # classifier
        pred = func.softmax(pred, dim=-1)
        return pred


if __name__ == '__main__':
    pass
