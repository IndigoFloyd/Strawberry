# Strawberry Segmentation and Classify
By YOLOv8m + ResUNet and YOLOv8m-seg models.\
Download models at https://pan.baidu.com/s/11jCi9lJ9jkfOOKCmeUPbZA?pwd=7ch9.
# 草莓成熟度分类 项目技术文档

## 项目背景

### 项目目的

将草莓根据红色区域占比分类，标准为：

草莓表面红色覆盖85%为成熟(ripe)；50%-85%为转色后期(Turning Late)；0.01%-50%为转色前期(Turning Early)；小于等于0.01%为青果(white)。

### 项目环境

#### 数据集来源

开源的草莓分割数据集，标签为png格式的蒙版图像，依据草莓的序号标记分割区域。[^1]

#### 环境配置

 CPU：Ryzen 9 5900HS

GPU：GeForce GTX 1650 with max-q design 4GB

RAM：16GB

CUDA：12.1

## 实现思路

<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src="MDimages/usecase.png"
         alt="无法显示图片时显示的文字"
         style="zoom:50%"/>
    <br>		<!--换行-->
    图1. 系统用例图示意	<!--标题-->
    </center>
</div>



### 图片输入和预处理

在模型训练时，为了更好地凸显目标特征，同时使得矩阵元素较小便于计算，对于图像进行标准化是必要的。将图像像素归一化至指定范围的公式如下：\
$`dst(r,c) = (b-a)*\frac{src(r, c)-min(src)}{max(src)-min(src)}+a`$\
使用OpenCV进行处理即为：

```python
cv2.normalize(cv2.normalize(self.img, normal, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))  # NORM_MINMAX即实现上述公式，此外需要将图像设置为浮点类型
```

除此之外，在训练过程前，也需要对于图像进行增强以一定程度上增强模型鲁棒性。在训练YOLOv8-seg模型时，数据集为原数据集（3100张）基础上，抽取的310张图像经过了加噪声（均值为0，标准差为5的高斯噪声）操作，抽取的775张图片进行了随机进行不等比缩放（0.5到2.5倍）的操作，此外为了进一步学习特征，还加入了7539张裁剪出的单个草莓的图像以及300张原始图像，按照70%、15%、15%的比例划分训练集、测试集、验证集。在训练U-Net模型时，采取了对裁剪出的单个草莓的图像进行旋转的操作（等可能地选取水平翻转、垂直翻转、水平垂直翻转与不翻转）。

### 检测和分割

该项目的关键之处就在于实例分割，如果简单地理解为语义分割任务，无法对单个草莓进行红色区域占比计算，而简单地理解为目标检测也无法精确地进行后续计算。根据理解，这一步应有两种实现方式：one-stage与two-stage。

one-stage即一步到位，同时进行锚框和像素级别的分割掩码的预测。为实现这种模式，我选用了YOLOv8-seg模型（M Size）作为一种实例分割的实现方法。

two-stage即顺次进行，先进行目标检测定位感兴趣区域（ROI），然后对目标区域进行语义分割。为实现这种模式，我选用了YOLOv8模型（同为M Size）来进行目标检测，识别出单个的草莓，进而使用U-Net这一经典的语义分割模型实现语义分割。为进一步提升性能，增强模型效果，我选用ResNet-34[^2]的结构替换U-Net[^3]原有的Encode部分，并使用残差块替换掉所有的双层卷积块，便于在更深层次的卷积中保持较好性能。

<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src="MDimages/model.png"
         alt="无法显示图片时显示的文字"
         style="zoom:100%"/>
    <br>		<!--换行-->
    图2. ResNet-UNet示意	<!--标题-->
    </center>
</div>



​		

### 红色区域计算

为计算红色区域占比，首先应将图片由RGB空间转换为HSV空间，并根据阈值来划分红色区域。这样就划分出了掩膜，再通过计算划分前后非0区域面积之比，即可得到红色区域的占比。使用OpenCV实现方法如下：

```python
hsv = cv2.cvtColor(berry, cv2.COLOR_BGR2HSV)
# 红色区域在H通道上有0-10, 165-180两段区域，需要分开计算
lower = np.array([0, 50, 50])
upper = np.array([10, 255, 255])
mask0 = cv2.inRange(hsv, lower, upper)
lower = np.array([165, 50, 50])
upper = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower, upper)
# 合并掩膜
mask = mask0 + mask1
if np.count_nonzero(berry) != 0:
    ratio = np.count_nonzero(mask) / np.count_nonzero(berry) * 3  # 乘3是因为有三个通道
```

### 成熟度分类

因为已经给定范畴，故设置一组判断语句即可。

## 组件功能

### forward.py

该文件用于模型的推理，相关函数已封装入Strawberry类，可以直接调用。

---------------------------------------------2023.8.13更新------------------------------------------------

目前已支持视频文件的输入，提供```video-segment```与```video-detect```两种```mode```进行视频文件的处理。模型并未改变。用例见[使用方法](#10.使用方法)。

 #### 1. \__init__()

该函数用于初始化类，其中```self.imgPath``` 为输入图像路径，```self.outPath```为输出图像路径，```mode```为两类模型的选择（若为“segment”则使用YOLOv8-seg，若为“detect”则使用YOLOv8+U-Net）。

#### 2. sigmoid()

该函数用于将输入内容经过sigmoid函数计算后输出结果。

#### 3. cut()

该函数用于将输入图像等比例裁剪，并放在一个640*640的黑色背景上。这样做的目的在于保持图像原有特征，并将其缩放到模型输入的大小。需要注意的是，如果图像和背景数据类型不匹配，则会只有黑和白两种颜色，因此在创建背景时需要写明```dtype=img.dtype```。

#### 4. preprocess()

该函数用于对输入图像进行归一化，值得注意的是在此处图像的数据类型变为浮点型。返回的是一个交换了R通道和B通道的blob，可以直接输入进YOLO的ONNX模型。

#### 5. detect()

该函数用于使用YOLOv8+U-Net的模型推理，其中```self.model1```为YOLO的目标检测模型，而```self.model2```为U-Net的语义分割模型。在该函数中首先使用ONNXRuntime进行推理，得到```outs```，其大小为(1, 1, 6, 8400)。其中6的前4个数字代表锚框的中心x、y坐标以及锚框的宽度w和高度h，由此可以推得锚框的左上角和右下角点坐标。后2个数字代表0（无实际意义）和1（Strawberry）两类的置信度。8400代表YOLO的锚框划分（80\*80+40\*40+20\*20=8400）后的框数。在根据```score```与阈值的关系判断后可以得到检测到草莓的坐标，之后利用PyTorch的nms（非极大值抑制）模块来消除冗余框，进而裁剪出每个草莓的图像

接下来，通过循环将每个草莓的图像进行遍历，经过归一化、变形、调整数据类型后输入进```self.model2```，得到输出```out```（64\*64），经过阈值筛选将输出转化为0-255的单通道掩膜，使用```cv2.bitwise_and```划分出每个草莓的RGB三通道图像。最后的输出是```resultDict```，其key为序号，value为第一个元素为锚框x0、y0、x1、y1坐标的列表，第二个元素为得到的单个草莓图像。

#### 6. segment()

该函数同样是为了得到```resultDict```，其过程主要区别在于YOLOv8-seg得到的输出有两个，一为(38, 8400)，一为(32, 160, 160)。38的前6个数字与```detect()```方法中一致，后32个数字为mask的参数，需要转置为(8400, 32)与后一矩阵相乘得到最终的mask得分，再经过```sigmoid()```与阈值处理得到0-255的掩膜图像，此后的操作与detect()相似，不做赘述。

#### 7. red()

该函数解析```resultDict```，将其中每个草莓的图像转换为HSV色彩空间，并基于阈值进行色彩掩膜生成，最终根据前后图像非0区域面积大小之比得到红色占比，进而进行分类。以标签（text）和矩形（rectangle）绘制在原图像```self.img```上。

#### 8. show()

该函数用于展示```self.img```。

#### 9. output()

该函数用于将结果图像先裁掉黑边，再输出到```self.outPath```。

#### 10. 使用方法

一种用例如下：

```python
time_total = 0  # 总用时
for _ in range(10):  # 进行10轮重复
    start = time.time()  # 设置起始计时器
    s = Strawberry(r"D:\Projects\Python\Strawberry\U7hW5B86sL.jpg", mode="detect")  # 实例化类
    s.red(s.detect())  # 计算比例
    s.output()  # 输出图像
    end = time.time()  # 设置终止计时器
    time_total += end-start  # 得到时间差值，累加进总用时
print(f"{time_total / 10:.2f}")  # 输出平均用时
```

---------------------------------------------2023.8.13更新------------------------------------------------

一种视频测试用例如下：

```python
time_total = 0
for _ in range(15):
    start = time.time()
    s = Strawberry(r"D:\Projects\Python\gitstrawberry\strawberry_cut.mp4", mode="video-seg")  #视频路径和模型
    #s = Strawberry(r"D:\Projects\Python\gitstrawberry\strawberry_cut.mp4", mode="video-detect")  #视频路径和模型
    end = time.time()
    time_total += end - start
    print(end - start)
print(f"{time_total / 15:.2f}")
```



### resnet18.py

该文件用于U-Net的搭建与训练、测试。

#### 1. ResBlock

该类继承自```nn.Module```，主要用于搭建单个的残差块，其主要结构参考[^3]。

#### 2. ResNet18

该类继承自```nn.Module```，主要用于构建残差块层，其中```makeLayer()```方法用于构建重复的残差块层，重复次数根据所需构建的模型类别（如ResNet-18、ResNet-34等等）而定。也用于替换U-Net原论文[^2 ] 的双层卷积。

#### 3. UNet

该类继承自```nn.Module```，主要用于搭建U-Net网络，其结构参考了U-Net原论文，在Encode部分，使用ResNet-34进行了替换，Decode部分也使用```ResBlock```进行替换。

#### 4. dataloader

该类继承自```torch.utils.data.Dataset```，主要用于构建数据读取器。其中主要特点在于写了```augment```方法，用于随机翻转图像以增强其鲁棒性。

#### 5. 使用方法

训练用例：

```python
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
    # 在训练集上
    for img, label in train:
        model.train()  # 设置模型为训练状态
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
    # 在验证集上
    for img, label in val:
        model.eval()  # 设置模型为评估状态
        outputs = model(img)
        loss_val = criterion(outputs, label)

    print(f"epoch【{epoch+1} / {epochs}】, loss: {loss.item():.4f}, loss on val: {loss_val.item():.4f}")
```

导出用例：

```python
model = UNet(1)
model.to('cuda')
# 随意创建一个输入
dummy_input = torch.randn(1, 3, 64, 64).to('cuda')
onnx_path = "model.onnx"
torch.onnx.export(model, dummy_input, onnx_path, verbose=True)
```

测试用例：

```python
model = UNet(1)
model.to('cuda')
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
```



### benchmark.py

该文件主要用于U-Net性能测试，提供了```Precision```和```Recall```两个评估指标。

#### 1. 使用方法

用例如下：

```python
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
```

## 结果展示

### 模型训练

<center class="half">
    <img src="MDimages/segment-results.png" width="400"/>
    <img src="MDimages/detect-results.png" width="250"/>
</center>
<center>图3.YOLOv8-seg（左）和YOLOv8（右）模型的训练结果</center>

最终，YOLOv8-seg模型在验证集上取得了0.9553（Precision）、0.8618（Recall）、0.9265（mAP50）的成绩，YOLOv8模型在验证集上取得了0.9452（Precision）、0.9258（Recall）、0.9530（mAP50）的成绩。ResNet34-UNet在验证集上取得了0.9131（Precision）、0.9257（Recall）的成绩。

### 预测效果

<center class="half">
    <img src="MDimages/output.png" width="300"/>
    <img src="MDimages/output1.png" width="300"/>
</center>
<center>图4.YOLOv8-seg（左）和YOLOv8+U-Net（右）模型的预测结果</center>

在10轮循环中，YOLOv8-seg运行的平均时长为5.35s，而YOLOv8+U-Net运行的平均时长为5.09s，速度提升约5%。

在预测的结果中，可以明显看出YOLOv8-seg的效果更好，正确地找到了所有的草莓，甚至包括左上角极小的草莓。而组合模型的效果则是要差一些，正面被叶子遮挡的草莓并没有准确识别到，左上角极小的草莓也没有识别到。

<center class="half">
<img src="MDimages/output2.png" width="300"/>
<img src="MDimages/output3.png" width="300"/>
    <br></br>
<img src="MDimages/output4.png" width="250"/>
<img src="MDimages/output5.png" width="250"/>
<center>图5.YOLOv8-seg（左）和YOLOv8+U-Net（右）模型的泛化预测结果</center>
</center>


为测试泛化性，我从网上下载了两张图片，可以看出，第一张与数据集环境较为接近，而第二张与数据集环境、形态差异都很大。就结果来看，第一张图片两种模型都成功识别并定位到草莓，但是对于红色区域比例的计算方面，YOLOv8+U-Net组合的效果明显更可靠。第二张图片YOLOv8-seg模型并没有识别到草莓，而组合模型虽然识别到但并不准确，由此看来，二者皆具备一定程度上的泛化能力，但受限于数据集规模，并没有更强的泛化性。YOLOv8+U-Net的组合模型在成熟度计算方面优势更大。综上推断，YOLOv8-seg模型在数据集上出现了一定的过拟合现象。

---------------------------------------------2023.8.13更新------------------------------------------------

<img src="MDimages/output_video.gif"/>
<center>图6.YOLOv8-seg（左）和YOLOv8+U-Net（右）模型的预测结果（视频限制帧率为10）</center>

测试视频节选自https://www.youtube.com/watch?v=wE-NyfIEQmA。同样考验了两个模型的泛化性强弱。

在15轮循环中，YOLOv8-seg的平均用时为76.77s，YOLOv8+U-Net的组合模型平均用时为15.39s，组合模型的实际帧数约为单独模型的5倍，在[测试平台](#环境配置)上，组合模型FPS约可达到5FPS。两个更加轻量化的模型组合起来，体现出了速度上的巨大优势。且从效果上来看，右侧的组合模型更加稳定准确。

### 结果分析

对于YOLOv8的目标检测效果而言，由于未对数据集进行过多处理，其识别性能并不很好，这也与我有意想让模型在数据集上出现一定程度的过拟合有关，我希望借助更多的ROI进一步筛选并用U-Net识别，但效果并不算特别好。值得思考的是，这种效果难道真的就更弱吗？视情况而定，当前的目标检测模型主要识别到的还是比较完整的且较大的草莓，这样计算出的结果当然会更加贴近实际，由于算法问题，无力为被遮挡的草莓轮廓进行填充，因此很可能会出现错误的分类——比如叶子遮挡住了未转色区域，就会导致红色占比偏高。这一问题或许需要用到GAN来进行模拟解决，但这样无疑舍本逐末，在工业上大大提高了运算复杂度，最终的效果也未必就是最好。所以只识别完整的草莓一定意义上反倒更能准确地进行判断。

至于泛化能力强弱，我想，农业应用场景本身就远复杂于其它图像处理场景如医疗影像分割等等，其多变性与不确定性很难说能够一个模型“一招鲜，吃遍天”。因此在某一给定环境上能够实现较好的效果已经足以满足工业应用。联想到前期进行的数次调研报告，植物工厂的又一重要意义就在于规范化了生产场景，将背景复杂度几乎降低到了最低程度，同时也可以利用机械臂等设施进行高通量的表型数据采集，这远比本次项目利用的人工手机拍摄数据集强得多。得数据者得天下，这是一个非常显然的道理。

最后，这次自己从头搭建一个网络还是比较复杂的，但当最终发现将原论文中的Encode部分替换掉，换为ResNet-34，并用残差块更换掉Decode部分的双层卷积得到的良好效果，不免令人兴奋。在同样采用BCE损失函数的基础上，相同的Epoch下修改过的Loss大致稳定在0.3附近，而未修改的则在0.5-0.6左右波动，这一提升也再次证明了深层网络中ResNet的优越性。在自己实现的two-stage与官方给出的one-stage的对比中，我也更加理解到了SOTA并非万能的，多尝试网络结构，才能找到最佳的解决方案，才能在各有利弊的结果中学到更多。

### 改进展望

正如上文所述，当前训练的数据集还有待提升，主要在于数据扩增手段（如旋转等的加入）的增多以及样本量的增加。此外，对于被遮挡的草莓，是否能够使用GAN预测其完整模型？甚至是否可以只用GAN来预测植物的生长情况？[^4]当前只针对于2D图像，未来是否可以使用深度相机进行三维重建以获得更完整的表型信息？

此外，在模型结构方面，再加入注意力机制、引入Transformer等或许也可为模型提供更强大的性能。在未来，该项目依旧有发展空间。



## 文献引用

[^1]: Pérez-Borrero, I., Marín-Santos, D., Gegúndez-Arias, M. E., & Cortés-Ancos, E. (2020). A fast and accurate deep learning method for strawberry instance segmentation. Computers and Electronics in Agriculture, 178, 105736. https://doi.org/https://doi.org/10.1016/j.compag.2020.105736
[^2]: 2 He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 770–778.
[^3]: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, 234–241.
[^4]: Lu, Y., Chen, D., Olaniyi, E., & Huang, Y. (2022). Generative adversarial networks (GANs) for image augmentation in agriculture: A systematic review. Computers and Electronics in Agriculture, 200, 107208.
