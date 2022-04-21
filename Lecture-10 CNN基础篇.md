Notes：

1. 卷积层：保留图像的空间信息

2. 全连接层 丧失了图片原有的空间信息

3. 卷积(convolution)后，C(Channels)变，W(width)和H(Height)可变可不变，取决于是否padding。subsampling(或pooling)后，C不变，W和H变。

4. 前一部分叫做Feature Extraction，后一部分叫做classification

5. 数乘：对应元素相乘。 区别于矩阵乘。

6. 每一个输入来的通道 配 一个卷积核通道，即 输入的通道数量 和你卷积核的通道数量是一致的。 最后计算后的加起来。

![](C:\pywork\learn\imgs\2022-04-10-18-01-32-image.png)

![](C:\pywork\learn\imgs\2022-04-10-18-04-23-image.png)

输入的图 通道是n， 卷积核的通道也一定是n，为了使 卷积操作后的 featuremap 通道是 m，则需要用m的卷积核。

每一个卷积核它的通道数量要求和输入通道是一样的。这种卷积核的总数有多少个和你输出通道的数量是一样的。

7. 卷积层要求输入输出是四维张量(B,C,W,H)，全连接层的输入与输出都是二维张量(B,Input_feature)。

8. pytorch 输入 必须小批量，所以 tensor 必须最前有一个 batch(样本数量)，即 B,C,W,H

9. self.fc = torch.nn.Linear(320, 10)，这个320获取的方式，可以通过x = x.view(batch_size, -1) # print(x.shape)可得到(64,320),64指的是batch，320就是指要进行全连接操作时，输入的特征维度。
