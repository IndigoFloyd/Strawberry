import os
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score
import onnxruntime as onnx


def calculate_precision_recall(predictions, targets):
    # Convert predictions and targets to binary masks
    pred_masks = predictions > 0.8  # Use a threshold here

    # Calculate precision and recall
    precision = precision_score(targets.flatten(), pred_masks.flatten())
    recall = recall_score(targets.flatten(), pred_masks.flatten())

    return precision, recall


# 导入模型
model = onnx.InferenceSession(r"D:\Projects\Python\Strawberry\models/YOLOV8m/model.onnx", providers=['CUDAExecutionProvider'])
# 总精度
precision = 0
# 总召回
recall = 0
# 遍历验证集进行计算
for name in os.listdir(r"F:\strawberry\masks\images\val"):
    img = cv2.imread(rf"F:\strawberry\masks\images\val\{name}")
    label = cv2.imread(rf"F:\strawberry\masks\labels\val\{name}", cv2.IMREAD_GRAYSCALE)
    cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
    img = np.reshape(img, (1, 3, 64, 64)).astype(np.float32)
    outs = model.run(['267'], {"input.1": img})[0][0][0]
    precision_, recall_ = calculate_precision_recall(outs, label)
    precision += precision_
    recall += recall_
# 输出召回率和精度
print("Precision:", f"{precision / len(os.listdir(r'F:/strawberry/masks/images/val')):.4f}")
print("Recall:", f"{recall / len(os.listdir(r'F:/strawberry/masks/images/val')):.4f}")

