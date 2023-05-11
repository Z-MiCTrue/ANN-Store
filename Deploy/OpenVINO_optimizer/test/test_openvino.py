from openvino.inference_engine import IECore
from openvino.runtime import Core
import numpy as np


if __name__ == '__main__':
    # device
    ie_core = IECore()
    print('devices: ', ie_core.available_devices)
    # run
    core = Core()
    net_small = core.compile_model(model='test_small.onnx', device_name="MYRIAD")  # CPU GPU MYRIAD ...
    output = net_small([np.array([1], dtype=np.float32)])[next(iter(net_small.outputs))]
    print('small model work state: ', output)
    net_bigger = core.compile_model(model='test_bigger.onnx', device_name="MYRIAD")  # CPU GPU MYRIAD ...
    output = net_bigger([np.zeros((2125, 33), dtype=np.float32)])[next(iter(net_bigger.outputs))]
    print('bigger model work state: ', output)
    print('over')
