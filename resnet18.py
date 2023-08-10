import torch
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import random
import onnxruntime


class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResBlock, self).__init__()       
        # 第一个3x3卷积核，padding为1不改变尺寸大小
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # 归一化
        self.bn1 = nn.BatchNorm2d(output_channels)
        # ReLu
        self.relu = nn.ReLU(inplace=True)
        # 第二个3x3卷积核
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # 归一化
        self.bn2 = nn.BatchNorm2d(output_channels)
        # 尺寸修正，经过上述两次卷积size会变为(ori-1) / stride + 1，如果stride != 1，与原先的输入尺寸不一样，就无法完成相加，所以要用等stride，且kernel size=1，padding=0的卷积核
        # 如果输入、输出通道数不同也同理，这样可以调整原始输入的通道数
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(input_channels, output_channels, stride=stride, kernel_size=1, bias=False), nn.BatchNorm2d(output_channels))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = x
        # 第一次卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 第二次卷积
        out = self.conv2(out)
        out = self.bn2(out)
        # 残差边计算
        out += self.shortcut(identity)
        # 激活
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes = 2):
        super(ResNet18, self).__init__()
        # 输入三通道图像
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 第一个残差层输出尺寸变小由最大池化导致，所以stride=1，不改变大小
        self.layer1 = self.makeLayer(64, 64, 2)
        # 第二个残差层输出尺寸缩小一半，stride=2
        self.layer2 = self.makeLayer(64, 128, 2, stride=2)
        # 同第二个残差层
        self.layer3 = self.makeLayer(128, 256, 2, stride=2)
        # 同第二个残差层
        self.layer4 = self.makeLayer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    # 便于搭建多个残差块层
    def makeLayer(self, input_channels, output_channels, times, stride=1):
        layer = []
        # 第一个的输入通道数不一样，所以单独拿出来
        layer.append(ResBlock(input_channels, output_channels, stride=stride))
        for _ in range(1, times):
            layer.append(ResBlock(output_channels, output_channels, stride=1))
        # 打包
        return nn.Sequential(*layer)

    def forward(self, x):
        # 第一次卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 最大池化
        out = self.maxpool(out)
        # 残差层
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 平均池化和全连接
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

class UNet(nn.Module):
    def __init__(self, num_classes = 2):
        super(UNet, self).__init__()
        res = ResNet18()
        self.conv1 = res.makeLayer(3, 64, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = res.makeLayer(64, 128, 3, 1)
        self.conv3 = res.makeLayer(128, 256, 4, 2)
        self.conv4 = res.makeLayer(256, 512, 6, 2)
        self.conv5 = res.makeLayer(512, 1024, 3, 2)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv6 = res.makeLayer(1024, 512, 1)
        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv7 = res.makeLayer(512, 256, 1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv8 = res.makeLayer(256, 128, 1)
        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv9 = res.makeLayer(128, 64, 1)
        self.conv10 = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encode
        out1 = self.conv1(x)
        pool1 = self.maxpool(out1)
        out2 = self.conv2(pool1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        # Decode
        up1 = self.upconv1(out5)
        cat1 = torch.cat((out4, up1), 1)
        out6 = self.conv6(cat1)
        up2 = self.upconv2(out6)
        cat2 = torch.cat((out3, up2), 1)
        out7 = self.conv7(cat2)
        up3 = self.upconv3(out7)
        cat3 = torch.cat((out2, up3), 1)
        out8 = self.conv8(cat3)
        up4 = self.upconv4(out8)
        cat4 = torch.cat((out1, up4), 1)
        out9 = self.conv9(cat4)
        out10 = self.conv10(out9)
        
        return torch.sigmoid(out10)

class dataloader(Dataset):
    def __init__(self, data_dir, datatype, transform=None):
        self.datalist = os.listdir(data_dir + f'/images/{datatype}')
        self.type = datatype
        self.data_dir = data_dir
        self.transform = transform
    
    def augment(self, image, flipcode):
        flip = cv2.flip(image, flipcode)
        return flip
    
    def __getitem__(self, index):
        # 获取图片名称
        imgname = self.datalist[index]
        # 读取图片和标签
        img = cv2.imread(f"{self.data_dir}/images/{self.type}/{imgname}")
        img = img
        cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
        label = cv2.imread(f"{self.data_dir}/labels/{self.type}/{imgname}", cv2.IMREAD_GRAYSCALE)
        # 翻转可能的列表（2为不执行翻转）
        flipcode = random.choice([-1, 0, 1, 2])
        if flipcode != 2:
            img = self.augment(img, flipcode)
            label = self.augment(label, flipcode)
        img = np.reshape(img, (3, 64, 64))
        img = torch.from_numpy(img).float().to('cuda')
        label = np.reshape(label, (1, 64, 64))
        label = torch.from_numpy(label).float().to('cuda')
        return img, label
    
    def __len__(self):
        return len(self.datalist)

model = UNet(1)
model.to('cuda')
# 训练部分
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
dataset_train = dataloader("/media/pfgaolinux/084E161F084E161F/strawberry/masks", 'train')
dataset_val = dataloader("/media/pfgaolinux/084E161F084E161F/strawberry/masks", 'val')
batchsize = 128
train = DataLoader(dataset_train, batch_size=batchsize, shuffle=True)
val = DataLoader(dataset_val, batch_size=batchsize, shuffle=False)
epochs = 60
for epoch in range(epochs):
    for img, label in train:
        model.train()
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
    for img, label in val:
        model.eval()
        outputs = model(img)
        loss_val = criterion(outputs, label)

    print(f"epoch【{epoch+1} / {epochs}, loss: {loss.item():.4f}, loss on val: {loss_val.item():.4f}")

# 导出为onnx模型
# dummy_input = torch.randn(1, 3, 64, 64).to('cuda')
# onnx_path = "model.onnx"
# torch.onnx.export(model, dummy_input, onnx_path, verbose=True)

# 测试部分=
model = onnxruntime.InferenceSession('model.onnx', providers=['CUDAExecutionProvider'])
img = cv2.imread("/media/pfgaolinux/084E161F084E161F/ultralytics/ultralytics/double.jpg")
cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
img = cv2.resize(img, (64, 64))
img = np.reshape(img, (1, 3, 64, 64)).astype(np.float32)
out = model.run(['267'], {"input.1": img})
_, test = cv2.threshold(out[0][0][0], 0.6, 255, cv2.THRESH_BINARY)
cv2.imshow('img', test)
cv2.waitKey(0)
cv2.destroyAllWindows()
