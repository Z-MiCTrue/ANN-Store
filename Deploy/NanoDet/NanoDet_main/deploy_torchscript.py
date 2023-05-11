import time

import numpy as np
import torch
import cv2

from post_process import (img_resize, generate_map, decode_inference, use_NMS, Streamer)


class TorchScript_Forward:
    def __init__(self, net_name):
        self.net = torch.jit.load(net_name)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = self.net.to(self.device)
        self.net.eval()
        print('Use device: ', self.device)

    def forward(self, img, normalize=None):
        img = np.ascontiguousarray(img)
        if normalize is not None:
            img = (img - normalize[0]) / normalize[1]
        tensor = np.transpose(np.expand_dims(img, axis=0), (0, 3, 1, 2))
        tensor = torch.from_numpy(tensor).to(torch.float32)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            tensor = self.net.forward(tensor)
        tensor = tensor.to('cpu')
        return tensor


class Options:
    def __init__(self):
        # ---base--- #
        self.classes_name = ['H']
        self.score_min = 0.5
        self.IoU_max = 0.35
        self.input_size = (224, 224)  # w, h
        data_mean = np.array([103.53, 116.28, 123.675])
        data_std = np.array([57.375, 57.12, 58.395])
        self.reg_max = 5
        self.strides = [8, 16, 32]
        # ---auto--- #
        self.num_classes = len(self.classes_name)
        self.normalize = [data_mean, data_std]
        self.FeatureMap_sizes = [[np.ceil(self.input_size[1] / stride), np.ceil(self.input_size[0] / stride)]
                                 for stride in self.strides]
        self.centre_maps = [np.tile(generate_map(self.FeatureMap_sizes[i], stride), (1, 2))
                            for i, stride in enumerate(self.strides)]


if __name__ == '__main__':
    options = Options()
    ts_model = TorchScript_Forward('nets/nanodet_ts.pt')
    test_img = False
    camera_id = 1
    if test_img:
        frame = cv2.imread('test.jpg', 1)
        frame = img_resize(frame, options.input_size, keep_ratio=True)
        # 推理
        predicts = np.squeeze(ts_model.forward(frame, options.normalize).numpy(), axis=0)
        # 解码
        predicts = decode_inference(predicts, options)
        # 非极大值抑制
        res_bbox = use_NMS(predicts, IoU_max=options.IoU_max)
        # 显示
        for each_bbox in res_bbox:
            # 画框
            cv2.rectangle(frame, tuple(each_bbox[:2].astype(np.int16)), tuple(each_bbox[2: 4].astype(np.int16)),
                          (255, 0, 0))
            # 获得最大概率类别索引
            class_index = np.argmax(each_bbox[4:])
            # 获得最大概率类别概率值
            class_possible = str(np.round(each_bbox[4:][class_index], 4))
            cv2.putText(frame, options.classes_name[class_index] + ' ' + class_possible,
                        tuple(each_bbox[:2].astype(np.int16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))
        cv2.imshow('frame', frame)
        k = cv2.waitKey()
    else:
        streamer = Streamer(camera_id)
        frame = streamer.grab_frame()
        fps = 0
        timestamp = time.time()
        while frame is not None:
            # 计算fps
            fps += 1
            now = time.time()
            if now - timestamp >= 1:
                timestamp = now
                print('fps:', '=' * fps, fps)
                fps = 0
            # 变形
            frame = img_resize(frame, options.input_size, keep_ratio=True)
            # 推理
            predicts = np.squeeze(ts_model.forward(frame, options.normalize).numpy(), axis=0)
            # 解码
            predicts = decode_inference(predicts, options)
            # 非极大值抑制
            res_bbox = use_NMS(predicts, IoU_max=options.IoU_max)
            for each_bbox in res_bbox:
                # 画框
                cv2.rectangle(frame, tuple(each_bbox[:2].astype(np.int16)), tuple(each_bbox[2: 4].astype(np.int16)),
                              (255, 0, 0))
                # 获得最大概率类别索引
                class_index = np.argmax(each_bbox[4:])
                # 获得最大概率类别概率值
                class_possible = str(np.round(each_bbox[4:][class_index], 4))
                cv2.putText(frame, options.classes_name[class_index] + ' ' + class_possible,
                            tuple(each_bbox[:2].astype(np.int16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))
            cv2.imshow('frame', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC out
                cv2.destroyAllWindows()
                break
            else:
                frame = streamer.grab_frame()
