import torch
from torch import nn

from net import LSTM


class My_lstm:
    def __init__(self, lr=1e-3, weight_decay=1e-4):
        # 设备状态
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Device: ', self.device)
        # 构建数据集
        self.X_train = None
        self.Y_train = None
        # 搭建网络
        self.ANN = LSTM(input_size=(3, 1, 5), hidden_size=(4, 8), output_size=(1, 1))
        self.ANN.device = self.device
        self.ANN.rebuild_hc()
        self.ANN = self.ANN.to(self.device)
        # 保存相关参数
        self.lr = lr  # 用于检查网络
        self.weight_decay = weight_decay  # 用于检查网络
        # 设置优化器and损失函数
        self.optimizer = torch.optim.Adam(self.ANN.parameters(), lr=lr, weight_decay=weight_decay)  # L2正则化
        self.loss_func = nn.MSELoss()  # reduction 维度有无缩减, 默认是 mean: 'none', 'mean', 'sum'

    def M_train(self, t_times=1e4, max_loss=0.):
        # 求解正则化参数
        X_train_ = torch.transpose(self.X_train, 1, 0).reshape(self.X_train.shape[1],
                                                               self.X_train.shape[0] * self.X_train.shape[2])
        self.ANN.mean = torch.unsqueeze(torch.unsqueeze(torch.mean(X_train_, dim=-1), dim=0), dim=-1).to(self.device)
        print('mean: ', self.ANN.mean)
        self.ANN.std = torch.unsqueeze(torch.unsqueeze(torch.std(X_train_, dim=-1), dim=0), dim=-1).to(self.device)
        print('std: ', self.ANN.std)
        # 将数据集转移至目设备
        self.X_train = self.X_train.to(self.device)
        self.Y_train = self.Y_train.to(self.device)
        # 开始迭代
        times_overflow = True
        for epoch in range(int(t_times)):
            out = self.ANN(self.X_train)
            loss = self.loss_func(out, self.Y_train)  # 计算误差
            if loss.item() < max_loss:  # torch.item()  用于提取张量为浮点数
                times_overflow = False
                break
            else:
                self.optimizer.zero_grad()  # 清除梯度
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新所有的参数
        if times_overflow:
            print('ANN Warning: Training Times Overflow')
        return times_overflow

    def Data_import(self, X_train, Y_train):
        self.X_train, self.Y_train = X_train, Y_train

    def Net_forward(self, X_pre):
        self.ANN.batch_size = 1
        self.ANN.rebuild_hc()
        # 将数据集转移至目设备
        X_pre = X_pre.to(self.device)
        with torch.no_grad():
            X_out = self.ANN(X_pre).to('cpu').data  # torch.data 为一种深拷贝方法
        return X_out


if __name__ == '__main__':
    x_train = torch.tensor([[[1, 2, 3, 4, 5]],
                            [[2, 3, 4, 5, 6]],
                            [[3, 4, 5, 6, 7]]], dtype=torch.float32)
    y_train = torch.tensor([[[6]], [[7]], [[8]]], dtype=torch.float32)
    x = torch.tensor([[[1.5, 2.5, 3.5, 4.5, 5.5]]], dtype=torch.float32)
    my_lstm = My_lstm(lr=1e-3, weight_decay=1e-4)
    # 导入数据集
    my_lstm.Data_import(x_train, y_train)
    # 训练数据集
    my_lstm.M_train(max_loss=1e-3)
    # 预测
    print(my_lstm.Net_forward(x))
