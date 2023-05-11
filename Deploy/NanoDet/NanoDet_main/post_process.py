from threading import Thread
from copy import deepcopy
import time

import numpy as np
import cv2


def func_sigmoid(z):
    return 1 / (1 + np.exp(-z))


def func_softmax(z):
    z_exp = np.exp(z)
    p = z_exp / np.expand_dims(np.sum(z_exp, axis=1), axis=-1)
    return p


#  缩放 size=(w, h)
def img_resize(img, size, keep_ratio=True, points=None):  # points:[[x, y]]
    h_ori, w_ori, channel = img.shape[:3]
    w_new, h_new = size
    # 需要补边(右下补)
    if keep_ratio and w_new / w_ori != h_new / h_ori:
        scale = min(w_new / w_ori, h_new / h_ori)
        w_valid, h_valid = round(w_ori * scale), round(h_ori * scale)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # 邻近区域的像素值平均作为填充值
        if channel == 1:
            if w_valid > h_valid:
                aim_size = np.full((h_new, w_new, channel), fill_value=np.mean(img[h_valid * 3 // 4:]),
                                   dtype=np.uint8)
            else:
                aim_size = np.full((h_new, w_new, channel), fill_value=np.mean(img[:, w_valid * 3 // 4:]),
                                   dtype=np.uint8)
        else:
            if w_valid > h_valid:
                aim_size = np.full((h_new, w_new, channel), fill_value=np.mean(img[h_valid * 3 // 4:],
                                                                               axis=(0, 1)), dtype=np.uint8)
            else:
                aim_size = np.full((h_new, w_new, channel), fill_value=np.mean(img[:, w_valid * 3 // 4:],
                                                                               axis=(0, 1)), dtype=np.uint8)
        aim_size[:h_valid, :w_valid] = img
        if points is None:
            return aim_size
        else:
            points = np.array(points)
            points = points * np.array([scale, scale, scale, scale])
            return aim_size, points.tolist()
    # 不需要改变
    elif w_new == w_ori and h_new == h_ori:
        if points is None:
            return img
        else:
            return img, points
    # 不需成比例或已成比例
    else:
        aim_size = cv2.resize(img, None, fx=w_new/w_ori, fy=h_new/h_ori, interpolation=cv2.INTER_AREA)
        if points is None:
            return aim_size
        else:
            fx = w_new / w_ori
            fy = h_new / h_ori
            points = np.array(points)
            points = points * np.array([fx, fy, fx, fy])
            return aim_size, points.tolist()


# 生成各特征图锚点相对原图坐标
def generate_map(map_size, stride):
    # map_size: (x, y); stride:int; return: [[0, 0], [0, stride], ...]
    relative_x, relative_y = np.meshgrid(np.arange(map_size[0]), np.arange(map_size[1]))
    map_area = int(map_size[0] * map_size[1])
    centre_map = (np.concatenate((np.expand_dims(relative_x, axis=-1),
                                  np.expand_dims(relative_y, axis=-1)), axis=-1) * stride).reshape(map_area, 2)
    return centre_map


# 将偏移概率转换为锚点偏移坐标
def offset_correction(bbox_pred, centre_map, stride, reg_max):
    points_count = deepcopy(bbox_pred.shape[0])
    bbox_pred = np.concatenate(np.split(bbox_pred, 4, axis=1), axis=0)
    # 加权偏移
    offset_location = np.array([-1, -1, 1, 1]).repeat(points_count) * np.dot(bbox_pred, np.arange(reg_max + 1).T).T
    offset_location = np.concatenate(np.split(np.expand_dims(offset_location, axis=-1), 4, axis=0), axis=1)
    real_location = centre_map + offset_location * stride
    return real_location  # return like 2125x4


def decode_inference(feature_map, opt, use_onnx_model=False):
    # 如果是onnx模型输出则不再需要将类别预测映射为概率
    if use_onnx_model:
        scores = feature_map[:, np.arange(opt.num_classes)]
    else:
        scores = func_sigmoid(feature_map[:, np.arange(opt.num_classes)])
    bbox_pred = feature_map[:, opt.num_classes:]
    # 将多分类输出映射为概率
    bbox_pred = np.concatenate(np.split(bbox_pred, 4, axis=1), axis=0)
    bbox_pred = func_softmax(bbox_pred)
    bbox_pred = np.concatenate(np.split(bbox_pred, 4, axis=0), axis=1)
    # 分割为每张特征尺度
    bbox_pred = np.split(bbox_pred,
                         np.cumsum(np.cumprod(opt.FeatureMap_sizes, axis=1)[:, -1])[:-1].astype(np.int32), axis=0)
    # 根据每张特征图计算, 循环次数为特征图数
    res = np.empty((0, 4))
    for i, stride in enumerate(opt.strides):
        # 获取中心点分布 [[x, y, x, y], ...]
        centre_map = opt.centre_maps[i]
        # 还原为坐标
        real_location = offset_correction(bbox_pred[i], centre_map, stride, opt.reg_max)
        # 添加
        res = np.append(res, real_location, axis=0)
    # 将预测类别数据与bbox数组合并
    res = np.concatenate((res, scores), axis=1)
    # 筛选置信度, 任意一类置信度超过阈值将被保留
    res = res[np.max(res[:, 4:], axis=1) >= opt.score_min]
    return res


def use_NMS(data_ori, IoU_max):
    # 按列分割 x_1, y_1, x_2, y_2, scores
    x_1, y_1, x_2, y_2 = np.split(data_ori[:, :4], 4, axis=1)
    scores = data_ori[:, 4:]
    # score_index_list是按照score最大值降序排序的索引列表
    score_index_list = np.argsort(-np.max(scores, axis=1), axis=0).flatten()
    # 每一个候选框的面积
    each_areas = (x_2 - x_1) * (y_2 - y_1)
    bbox_keep = []
    while score_index_list.size > 0:
        # 当前置信度最高加入保留组
        score_index = score_index_list[0]
        bbox_keep.append(score_index)
        # 计算当前概率矩形框与其他矩形框的相交框的坐标->得到的是数组(1-n维逐个比较)
        x_overlay_1 = np.maximum(x_1[score_index], x_1[score_index_list[1:]])
        y_overlay_1 = np.maximum(y_1[score_index], y_1[score_index_list[1:]])
        x_overlay_2 = np.minimum(x_2[score_index], x_2[score_index_list[1:]])
        y_overlay_2 = np.minimum(y_2[score_index], y_2[score_index_list[1:]])
        # 计算相交框的面积, 边长为负时用0代替
        overlay_w = np.maximum(0, x_overlay_2 - x_overlay_1)
        overlay_h = np.maximum(0, y_overlay_2 - y_overlay_1)
        area_overlay = overlay_w * overlay_h
        # 计算重叠度IOU：重叠面积 / (面积1 + 面积2 - 重叠面积)
        IoU = area_overlay / (each_areas[score_index] + each_areas[score_index_list[1:]] - area_overlay)
        # 找到重叠度低于阈值的矩形框索引并生成下一索引(除去当前+1)
        next_index = np.where(IoU <= IoU_max)[0] + 1
        # 将score_index_list序列更新, 仅保留过限bbox
        score_index_list = score_index_list[next_index]
    return data_ori[bbox_keep]


class Streamer:
    def __init__(self, cam_id):
        # def vc
        self.vc = cv2.VideoCapture(cam_id)
        # init status
        self.cam_state = False
        self.frame = None
        # creat threading
        self.thread = Thread(name='camera', target=self.update, daemon=True)  # open threading till main killed
        self.thread.start()
        print('camera threading start')
        # wait for camera to open
        time.sleep(2)

    def update(self):
        if self.vc.isOpened():
            self.cam_state = True
        while self.cam_state:
            self.cam_state, self.frame = self.vc.read()

    def grab_frame(self):
        if self.cam_state:
            return self.frame
        else:
            return None
