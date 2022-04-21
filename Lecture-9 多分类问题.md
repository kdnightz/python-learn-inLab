# Lecture-9 多分类问题

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F
import torch.optim as optim


batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))  # 均值， 标准差
])

train_dataset = datasets.MNIST(root='../dataset/mnist',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='../dataset/mnist',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)  # in_feature, out_feature 输入，输出为几维
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # 改变形状，成为矩阵
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()  # 取值，不然会构建计算图 data返回Tensor， .item()返回标量
        if batch_idx % 300 == 299:  # 每300次迭代 输出一次
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # test 不需要反向传播，只需要看正向，看分类算对了多少
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # data=1表示沿着第一个未维度，行，去找最大值， 返回 每一行的最大值，和最大值的下标
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        if epoch % 10 == 9:  # 每10轮测试一次
            test()
```

notes:

1. Dataloader： num_workers 多线程数量 win下不要用

2. 最后一层不做激活， 因为 torch自带的crossentropyloss() 已经有了。 softmax+log + (-Ylog(y_hat)) + ont-hot(独热)
   注： y需是 LongTensor()

3. PIL opencv 传入图片是 W*H*C 的 要转变为 C*W*H 在神经网络torch中处理更高效

4. <mark>x = x.view(-1, 784) # -1表示自动计算</mark>

5. softmax的输入不需要再做非线性变换，也就是说softmax之前不再需要激活函数(relu)。softmax两个作用，如果在进行softmax前的input有负数，通过指数变换，得到正数。所有类的概率求和为1。

6. y的标签编码方式是one-hot。我对one-hot的理解是只有一位是1，其他位为0。(但是标签的one-hot编码是算法完成的，算法的输入仍为原始标签)

7. 多分类问题，标签y的类型是LongTensor。比如说0-9分类问题，如果y = torch.LongTensor([3])，对应的one-hot是[0,0,0,1,0,0,0,0,0,0].(这里要注意，如果使用了one-hot，标签y的类型是LongTensor，糖尿病数据集中的target的类型是FloatTensor)

8. CrossEntropyLoss <==> LogSoftmax + NLLLoss。也就是说使用CrossEntropyLoss最后一层(线性层)是不需要做其他变化的；使用NLLLoss之前，需要对最后一层(线性层)先进行SoftMax处理，再进行log操作。

9. torch.max的返回值有两个，第一个是每一行的最大值是多少，第二个是每一行最大值的下标(索引)是多少。
