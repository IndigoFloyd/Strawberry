import cv2
import numpy as np
import onnxruntime as onnx
import torch
from torchvision.ops import nms
import time


class Strawberry:
    def __init__(self, imgPath, outPath='./', mode="segment"):
        self.imgPath = imgPath
        self.outPath = outPath
        self.img = cv2.imread(self.imgPath)
        # 用于最终裁剪黑色部分
        self.finalsize = (0, 0)
        # hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        # h_shift = np.random.uniform(-0.08, 0.08)
        # hsv_img[:, :, 0] = np.clip(hsv_img[:, :, 0] + h_shift, 0, 179)
        # self.img = hsv_img
        # self.img = cv2.GaussianBlur(self.img, (15, 15), 5)
        if mode == "segment":
            self.model = onnx.InferenceSession(r"D:\Projects\Python\Strawberry\models/YOLOV8m-seg/last.onnx", providers=['CUDAExecutionProvider'])
        elif mode == "detect":
            self.model1 = onnx.InferenceSession(r"D:\Projects\Python\Strawberry\models/YOLOV8m/last_detect.onnx", providers=['CUDAExecutionProvider'])
            self.model2 = onnx.InferenceSession(r"D:\Projects\Python\Strawberry\models/YOLOV8m/model.onnx", providers=['CUDAExecutionProvider'])

    def sigmoid(self, n):
        return 1 / (1 + np.exp(-n))

    def cut(self, img):
        # 标准化裁剪并转换为正常RGB
        if img.shape[0] >= img.shape[1]:
            scale = img.shape[1] / img.shape[0]
            width = int(640 * scale)
            reshaped = cv2.resize(img, (width, 640))
            self.finalsize = (width, 640)
            black = np.zeros((640, 640 - width, 3), dtype=img.dtype)
            img = np.hstack((reshaped, black))
        else:
            scale = img.shape[0] / img.shape[1]
            height = int(640 * scale)
            reshaped = cv2.resize(img, (640, height))
            self.finalsize = (640, height)
            black = np.zeros((640 - height, 640, 3), dtype=img.dtype)
            img = np.vstack((reshaped, black))
        return img

    def preprocess(self):
        # 原始图片归一化
        normal = np.zeros(self.img.shape, dtype=np.float32)
        cv2.normalize(self.img, normal, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        normal = self.cut(normal)
        self.img = self.cut(self.img)
        blob = cv2.dnn.blobFromImage(normal, swapRB=True)
        return blob

    def detect(self):
        blob = self.preprocess()
        outs = self.model1.run(["output0"], {"images": blob})
        output = outs[0][0]
        bboxes = []
        scores = []
        for i in range(len(output[0])):
            score = output[5, i]
            if score > 0.1:
                cx = output[0, i]
                cy = output[1, i]
                w = output[2, i]
                h = output[3, i]
                lx = int(int(cx - 0.5 * w) / 640 * self.img.shape[1])
                ly = int(int(cy - 0.5 * h) / 640 * self.img.shape[0])
                rx = int(int(cx + 0.5 * w) / 640 * self.img.shape[1])
                ry = int(int(cy + 0.5 * h) / 640 * self.img.shape[0])
                bboxes.append([lx, ly, rx, ry])
                scores.append(score)
        if len(bboxes) > 0:
            # 锚框坐标转换为张量
            bboxes_tensor = torch.Tensor(bboxes)
            # 得分转换为张量
            scores_tensor = torch.Tensor(scores)
            # 得到下标
            indices = nms(bboxes_tensor, scores_tensor, 0.5)
            resultDict = {}
            n = -1
            for i in indices:
                n += 1
                box = bboxes[i]
                if box[2] - box[0] > 15 and box[3] - box[1] > 15:
                    cut = self.img[box[1]: box[3], box[0]: box[2]]
                    cut_norm = np.zeros(cut.shape)
                    cv2.normalize(cut, cut_norm, 0, 1, cv2.NORM_MINMAX)
                    cut_reshaped = cv2.resize(cut_norm, (64, 64))
                    cut_reshaped = np.reshape(cut_reshaped, (1, 3, 64, 64)).astype(np.float32)
                    out = self.model2.run(['267'], {"input.1": cut_reshaped})[0][0][0]  # 64*64
                    mask = (out > 0.8).astype(np.uint8) * 255
                    mask = cv2.resize(mask, (cut.shape[1], cut.shape[0]))
                    berry = cv2.bitwise_and(cut, cut, mask=mask)
                    resultDict[i] = [box, berry]
        else:
            resultDict = False
        return resultDict

    def segment(self):
        blob = self.preprocess()
        # 运行模型得到output0和output1，其中output0的前6行分别为cx, cy, w, h, 类别0得分, 类别1得分
        outs = self.model.run(["output0", "output1"], {"images": blob})
        # 得到output0，也就是锚框部分，后面32行用于分割
        output0 = outs[0][0].transpose()  # 21504 * 38 (1024/8=128, 128*128+64*64+32*32=21504)
        boxes = output0[:, 0:6]  # 21504 * 6 (cx, cy, w, h, class0, class1)
        masks_para = output0[:, 6:]  # 掩膜参数
        # 得到output1，用于分割
        output1 = outs[1][0]  # 32 * 256 * 256
        output1_reshape = output1.reshape(output1.shape[0], output1.shape[1] * output1.shape[2])  # 32 * (256*256)
        # 相乘得到masks
        masks = np.dot(masks_para, output1_reshape)  # 21504 * 65536（21504个框和对应掩码）
        output = np.hstack((boxes, masks))
        # 遍历得分大于0.85的，画框
        bboxes = []
        scores = []
        maskList = []
        for i in range(len(output)):
            score = output[i, 5]
            if score > 0.45:
                cx = output[i, 0]
                cy = output[i, 1]
                w = output[i, 2]
                h = output[i, 3]
                lx = int(int(cx - 0.5 * w) / 640 * self.img.shape[1])
                ly = int(int(cy - 0.5 * h) / 640 * self.img.shape[0])
                rx = int(int(cx + 0.5 * w) / 640 * self.img.shape[1])
                ry = int(int(cy + 0.5 * h) / 640 * self.img.shape[0])
                mask = output[i, 6:].reshape(160, 160)
                bboxes.append([lx, ly, rx, ry])
                scores.append(score)
                maskList.append(mask)
                # cv2.rectangle(self.img, (lx, ly), (rx, ry), (255, 0, 0), 2)
        if len(bboxes) > 0:
            # 锚框坐标转换为张量
            bboxes_tensor = torch.Tensor(bboxes)
            # 得分转换为张量
            scores_tensor = torch.Tensor(scores)
            # 得到下标
            indices = nms(bboxes_tensor, scores_tensor, 0.5)
            resultDict = {}
            n = -1
            for i in indices:
                n += 1
                mask = maskList[i]
                # 由于是点属于物体的概率，所以需要sigmoid来矫正到0-1的范围
                mask_sigmoid = self.sigmoid(mask)
                # True变为1，False变为0，乘以255变为白色
                mask = (mask_sigmoid > 0.55).astype('uint8') * 255
                mask_lx = int(bboxes[i][0] / self.img.shape[1] * 160)
                mask_ly = int(bboxes[i][1] / self.img.shape[0] * 160)
                mask_rx = int(bboxes[i][2] / self.img.shape[1] * 160)
                mask_ry = int(bboxes[i][3] / self.img.shape[0] * 160)
                # mask = mask[mask_ly:mask_ry, mask_lx:mask_rx]
                # kernel_open = np.ones((3, 3), np.uint8)
                # kernel_close = np.ones((3, 3), np.uint8)
                # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
                # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
                mask_img = cv2.resize(mask, ((bboxes[i][2] - bboxes[i][0]), (bboxes[i][3] - bboxes[i][1])))
                # 构建与原图像等大的黑色背景
                background = np.zeros(self.img.shape[:2], dtype=np.uint8)
                # 构建mask
                background[bboxes[i][1]:bboxes[i][3], bboxes[i][0]:bboxes[i][2]] = mask_img
                # 根据mask取出草莓图像
                berry = cv2.bitwise_and(self.img, self.img, mask=background)
                # cv2.imshow('test', berry)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # 保存到字典，其中n作为key表示是第几个草莓，后面列表保存了对应的锚框位置和掩码信息
                resultDict[n] = [bboxes[i], berry]
                # cv2.rectangle(self.img, (bboxes[i][0], bboxes[i][1]), (bboxes[i][2], bboxes[i][3]), (255, 0, 0), 2)
            print(len(resultDict))
            return resultDict
        else:
            return False

    def red(self, resultDict):
        resultDict = resultDict
        if resultDict:
            for key, value in resultDict.items():
                berry = value[1]
                rec = value[0]
                hsv = cv2.cvtColor(berry, cv2.COLOR_BGR2HSV)
                lower = np.array([0, 50, 50])
                upper = np.array([10, 255, 255])
                mask0 = cv2.inRange(hsv, lower, upper)
                lower = np.array([165, 50, 50])
                upper = np.array([180, 255, 255])
                mask1 = cv2.inRange(hsv, lower, upper)
                mask = mask0 + mask1
                if np.count_nonzero(berry) != 0:
                    ratio = np.count_nonzero(mask) / np.count_nonzero(berry) * 3
                    if ratio >= 0.85:
                        cv2.rectangle(self.img, (rec[0], rec[1]), (rec[2], rec[3]), (114, 128, 250), 2)
                        cv2.putText(self.img, f"ripe: {ratio:.2f}", (int(rec[0] * 1.05), int(rec[1] * 1.05)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (114, 128, 250), 2)
                    elif ratio >= 0.5:
                        cv2.rectangle(self.img, (rec[0], rec[1]), (rec[2], rec[3]), (79, 165, 255), 2)
                        cv2.putText(self.img, f"Turning Late: {ratio:.2f}", (int(rec[0] * 1.05), int(rec[1] * 1.05)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (79, 165, 255), 2)
                    elif ratio >= 0.01:
                        cv2.rectangle(self.img, (rec[0], rec[1]), (rec[2], rec[3]), (250, 206, 135), 2)
                        cv2.putText(self.img, f"Turning Early: {ratio:.2f}", (int(rec[0] * 1.05), int(rec[1] * 1.05)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 206, 135), 2)
                    else:
                        cv2.rectangle(self.img, (rec[0], rec[1]), (rec[2], rec[3]), (208, 224, 64), 2)
                        cv2.putText(self.img, f"white: {ratio:.2f}", (int(rec[0] * 1.05), int(rec[1] * 1.05)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (208, 224, 64), 2)

    def show(self):
        cv2.imshow('test', self.img)
        cv2.namedWindow('test', 0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def output(self):
        out = self.img[:self.finalsize[1], :self.finalsize[0]]
        cv2.imwrite(f"{self.outPath}/output5.png", out)

time_total = 0
for _ in range(1):
    start = time.time()
    s = Strawberry(r"D:\Projects\Python\Strawberry\U7hW5B86sL.jpg", mode="detect")
    s.red(s.detect())
    s.output()
    end = time.time()
    time_total += end-start
print(f"{time_total:.2f}")
