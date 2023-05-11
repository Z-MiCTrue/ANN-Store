import torch
import torch.nn as nn
import torch.nn.functional as Func


class ANN_Net(nn.Module):
    def __init__(self, IO_num):
        super(ANN_Net, self).__init__()
        self.layer_o = nn.Linear(IO_num, 1)

    def forward(self, x_in):
        # In
        x_out = Func.relu(self.layer_o(x_in))
        return x_out


if __name__ == '__main__':
    x = torch.tensor([1]).to(torch.float32)
    net = ANN_Net(1)
    net.eval()
    # onnx导出
    torch.onnx.export(net, x, 'test.onnx',
                      verbose=True, keep_initializers_as_inputs=True, opset_version=11,
                      input_names=["input"], output_names=["output"])
    print("finished exporting onnx ")
    print(net(x))

