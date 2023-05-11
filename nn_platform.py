import os
from copy import deepcopy

import numpy as np
import torch

from io_function import save_h5, read_h5


class NN_Platform:
    def __init__(self, model, params):
        # 设备状态
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {self.device}')
        # 构建数据集
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        # 搭建网络
        self.ANN = model(params)
        # 正则化参数
        self.mean = 0
        self.std = 1
        # 保存相关参数
        self.lr = params.lr
        self.weight_decay = params.weight_decay
        self.min_loss = torch.inf
        self.train_log = []
        # 设置优化器and损失函数
        self.optimizer = torch.optim.Adam(self.ANN.parameters(), lr=self.lr, weight_decay=self.weight_decay)  # L2正则化
        self.loss_func = params.loss_func
        # 创建数据路径
        if not os.path.exists(params.net_data_dir):
            os.makedirs(params.net_data_dir)

    def net_load(self, params, norm_dataset=False):
        group_list, data_mat_list = read_h5(params.net_load_norm)
        self.mean = torch.from_numpy(data_mat_list[group_list.index('mean')].astype(np.float32))
        self.std = torch.from_numpy(data_mat_list[group_list.index('std')].astype(np.float32))
        self.ANN.load_state_dict(torch.load(params.net_load_weight, map_location=self.device))
        # 标准化数据集
        if norm_dataset:
            self.x_train = (self.x_train - self.mean) / self.std
            self.x_val = (self.x_val - self.mean) / self.std
        print('net load over')

    def data_import(self, x_train, y_train, val_size):
        # 划分数据集
        self.x_train, self.y_train = deepcopy(x_train[val_size:]), deepcopy(y_train[val_size:])
        self.x_val, self.y_val = deepcopy(x_train[:val_size]), deepcopy(y_train[:val_size])

    def data_normalize(self, params, channel_dim=1):
        x_train_reshape = torch.transpose(self.x_train, channel_dim, 0).reshape(self.x_train.shape[channel_dim], -1)
        std_shape = [1] * len(self.x_train.shape)
        std_shape[channel_dim] = self.x_train.shape[channel_dim]
        self.mean = torch.mean(x_train_reshape, dim=-1).reshape(std_shape)
        self.std = torch.std(x_train_reshape, dim=-1).reshape(std_shape)
        # std 有 0 时替换为 1
        nan_index = self.std == 0
        self.mean[nan_index] = 0
        self.std[nan_index] = 1
        print(f'mean: {self.mean.flatten()} \nstd: {self.std.flatten()}')
        # 标准化数据集
        if self.x_train is not None and self.x_val is not None:
            self.x_train = (self.x_train - self.mean) / self.std
            self.x_val = (self.x_val - self.mean) / self.std
        # 记录数据
        group_list = ['mean', 'std']
        data_mat_list = [self.mean.numpy(), self.std.numpy()]
        save_h5(filename=f'{params.net_data_dir}/norm.h5', group_list=group_list, data_mat_list=data_mat_list)

    def start_train(self, params):
        # 清除记录
        if self.min_loss != torch.inf:
            print('reset log')
            self.min_loss = torch.inf
            self.train_log = []
        # 将网络转移至目标设备
        self.ANN = self.ANN.to(self.device)
        self.ANN.train()  # 模式转换
        self.x_val = self.x_val.to(self.device)
        self.y_val = self.y_val.to(self.device)
        # 取得对应 batch_size 次数
        if self.x_train.shape[0] % params.batch_size:
            batch_number = self.x_train.shape[0] // params.batch_size + 1
        else:
            batch_number = self.x_train.shape[0] // params.batch_size
        # 开始迭代
        for epoch in range(int(params.t_times + 1)):
            for batch in range(batch_number):
                # 将数据转移至目标设备
                x_train = self.x_train[params.batch_size * batch: params.batch_size * (batch + 1)].to(self.device)
                y_train = self.y_train[params.batch_size * batch: params.batch_size * (batch + 1)].to(self.device)
                # 计算损失
                out = self.ANN(x_train)
                loss = self.loss_func(out, y_train)  # 计算误差
                # 损失反向传播
                self.optimizer.zero_grad()  # 清除梯度
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新所有的参数
                # 检查点
                if epoch % (params.t_times // 10) == 0:
                    print('#-----message-----#')
                    # 验证
                    self.ANN.eval()  # 模式转换
                    with torch.no_grad():
                        val = self.ANN(self.x_val)
                        val_loss = self.loss_func(val, self.y_val).item()  # 计算误差
                    self.ANN.train()  # 模式转换
                    # 打印日志
                    self.train_log.append([loss.item(), val_loss])  # train loss, val_loss
                    print(f'epoch: {epoch} [batch: {batch}] \nval loss: {val_loss}; train loss: {loss.item()}')
                    # 保存模型
                    if val_loss < self.min_loss:
                        self.min_loss = val_loss
                        torch.save(self.ANN.state_dict(), f'{params.net_data_dir}/weight.pth')
                        print('model saved over')
                    print('#-----------------#')
        # show
        print(f'the best model loss is: {self.min_loss}')
        return self.min_loss

    def net_forward(self, X_pre):
        X_pre = (X_pre - self.mean) / self.std
        # 将数据集转移至目设备
        self.ANN = self.ANN.to(self.device)
        self.ANN.eval()  # 模式转换
        X_pre = X_pre.to(self.device)
        with torch.no_grad():
            X_out = self.ANN(X_pre).to('cpu')
        return X_out

    # export onnx
    def onnx_export(self, params):
        import onnx
        import onnxsim
        # 构建输入样例
        dummy_input = torch.randn(params.inference_shape,
                                  dtype=torch.float32, requires_grad=True, device=torch.device('cpu'))
        # onnx导出
        torch.onnx.export(self.ANN.to('cpu'), dummy_input, f'{params.net_data_dir}/model.onnx',
                          verbose=True, keep_initializers_as_inputs=True, opset_version=11,
                          input_names=["input"], output_names=["output"])
        print("finished exporting onnx ")
        # onnx简化
        print("start simplifying onnx ")
        input_data = {"data": dummy_input.detach().cpu().numpy()}
        model_sim, flag = onnxsim.simplify(f'{params.net_data_dir}/model.onnx', input_data=input_data)
        if flag:
            onnx.save(model_sim, f'{params.net_data_dir}/model.onnx')
            print("simplify onnx successfully")
        else:
            print("simplify onnx failed")


if __name__ == '__main__':
    pass
