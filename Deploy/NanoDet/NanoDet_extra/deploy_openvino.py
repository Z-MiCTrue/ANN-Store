import time

import cv2
import numpy as np

from post_process import (img_resize, use_NMS, Streamer)


class OpenVINO_Forward:
    def __init__(self, net_path):
        if type(net_path) == str:
            from openvino.runtime import Core
            ie = Core()
            use_device = "CPU"  # CPU MYRIAD
            self.net = ie.compile_model(model=net_path, device_name=use_device)
            self.input_net = None
        else:
            from openvino.inference_engine import IECore
            ie = IECore()
            use_device = "CPU"  # CPU MYRIAD
            IR_net = ie.read_network(model=net_path[0], weights=net_path[1])
            self.net = ie.load_network(network=IR_net, device_name=use_device)
            self.input_net = next(iter(self.net.input_info))
        self.output_net = next(iter(self.net.outputs))

    def forward(self, img, normalize=None, score_min=0.5):
        img = np.ascontiguousarray(img)
        if normalize is not None:
            img = (img - normalize[0]) / normalize[1]
        if self.input_net is None:
            img = np.transpose(np.expand_dims(img, axis=0), (0, 3, 1, 2)).astype(np.float32)
            output = self.net([img])[self.output_net]
        else:
            img = np.transpose(np.expand_dims(img, axis=0), (0, 3, 1, 2)).astype(np.float16)
            output = self.net.infer(inputs={self.input_net: img})[self.output_net]
        output = output[np.max(output[:, 4:], axis=1) >= score_min]
        return output


class Options:
    def __init__(self):
        # ---base--- #
        self.input_size = (320, 320)  # w, h
        self.classes_name = ['R3', 'B3', 'R0', 'B0', 'R4', 'B4', 'land']
        self.score_min = 0.3
        self.IoU_max = 0.35
        data_mean = np.array([103.53, 116.28, 123.675])
        data_std = np.array([57.375, 57.12, 58.395])
        # ---auto--- #
        self.normalize = [data_mean, data_std]


if __name__ == '__main__':
    options = Options()
    openvino_model = OpenVINO_Forward(('nets/nanodet_fp16.xml', 'nets/nanodet_fp16.bin'))
    # openvino_model = OpenVINO_Forward('nets/nanodet.onnx')
    test_img = False
    camera_id = 0
    if test_img:
        frame = cv2.imread('test.jpg', 1)
        frame = img_resize(frame, options.input_size, keep_ratio=True)
        # 推理+解码
        predicts = openvino_model.forward(frame, options.normalize, score_min=options.score_min)
        # 非极大值抑制
        res_bbox = use_NMS(predicts, IoU_max=options.IoU_max)
        # 显示
        for each_bbox in res_bbox:
            # 画框
            cv2.rectangle(frame, tuple(each_bbox[:2].astype(np.int16)), tuple(each_bbox[2: 4].astype(np.int16)),
                          (255, 0, 0))
            # 获得最大概率类别索引
            class_index = int(np.argmax(each_bbox[4:]))
            # 获得最大概率类别概率值
            class_possible = str(np.round(each_bbox[4:][class_index], 4))
            cv2.putText(frame, options.classes_name[class_index] + ' ' + class_possible,
                        tuple(each_bbox[:2].astype(np.int16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (22, 7, 207))
        cv2.imshow('frame', frame)
        k = cv2.waitKey()
    else:
        streamer = Streamer(camera_id)
        frame = streamer.grab_frame()
        fps = 0
        timestamp = time.time()
        while frame is not None:
            # time.sleep(0.05)
            # 计算fps
            fps += 1
            now = time.time()
            if now - timestamp >= 1:
                timestamp = now
                print('fps:', '=' * fps, fps)
                fps = 0
            # 变形
            frame = img_resize(frame, options.input_size, keep_ratio=True)
            # 推理+解码
            predicts = openvino_model.forward(frame, options.normalize, score_min=options.score_min)
            # 非极大值抑制
            res_bbox = use_NMS(predicts, IoU_max=options.IoU_max)
            for each_bbox in res_bbox:
                # 画框
                cv2.rectangle(frame, tuple(each_bbox[:2].astype(np.int16)), tuple(each_bbox[2: 4].astype(np.int16)),
                              (255, 0, 0))
                # 获得最大概率类别索引
                class_index = int(np.argmax(each_bbox[4:]))
                # 获得最大概率类别概率值
                class_possible = str(np.round(each_bbox[4:][class_index], 4))
                cv2.putText(frame, options.classes_name[class_index] + ' ' + class_possible,
                            tuple(each_bbox[:2].astype(np.int16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255))
            cv2.imshow('frame', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC out
                cv2.destroyAllWindows()
                break
            else:
                frame = streamer.grab_frame()
