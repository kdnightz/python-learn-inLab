### PyTorch 中文文档 - torch

1. ***from_numpy***
   
   > torch.from_numpy(ndarray) → Tensor
   
     Numpy桥，将`numpy.ndarray` 转换为pytorch的 `Tensor`。 返回的张量tensor和numpy的ndarray共享同一内存空间。修改一个会导致另外一个也被修改。返回的张量不能改变大小。
   
     例子:
   
   ```python
   >>> a = numpy.array([1, 2, 3])
   >>> t = torch.from_numpy(a)
   >>> t
   torch.LongTensor([1, 2, 3])
   >>> t[0] = -1
   >>> a
   array([-1,  2,  3])
   ```

2. **torch.range**

> torch.range(start, end, step=1, out=None) → Tensor

    返回一个1维张量，有 floor((end−start)/step)+1 个元素。包含在半开区间`[start, end）`从`start`开始，以`step`为步长的一组值。 `step` 是两个值之间的间隔，即 xi+1=xi+step

**警告**：建议使用函数 `torch.arange()`

参数:

- start (float) – 序列的起始点
- end (float) – 序列的最终值
- step (int) – 相邻点的间隔大小
- out (Tensor, optional) – 结果张量

例子：

```python
>>> torch.range(1, 4)

 1
 2
 3
 4
[torch.FloatTensor of size 4]

>>> torch.range(1, 4, 0.5)

 1.0000
 1.5000
 2.0000
 2.5000
 3.0000
 3.5000
 4.0000
[torch.FloatTensor of size 7]
```

3. ### torch.cat
   
   > torch.cat(inputs, dimension=0) → Tensor

   在给定维度上对输入的张量序列`seq` 进行连接操作。

`torch.cat()`可以看做 `torch.split()` 和 `torch.chunk()`的反操作。 `cat()` 函数可以通过下面例子更好的理解。

参数:

- inputs (*sequence of Tensors*) – 可以是任意相同Tensor 类型的python 序列
- dimension (*int*, *optional*) – 沿着此维连接张量序列。

例子：

```python
>>> x = torch.randn(2, 3)
>>> x

 0.5983 -0.0341  2.4918
 1.5981 -0.5265 -0.8735
[torch.FloatTensor of size 2x3]

>>> torch.cat((x, x, x), 0)

 0.5983 -0.0341  2.4918
 1.5981 -0.5265 -0.8735
 0.5983 -0.0341  2.4918
 1.5981 -0.5265 -0.8735
 0.5983 -0.0341  2.4918
 1.5981 -0.5265 -0.8735
[torch.FloatTensor of size 6x3]

>>> torch.cat((x, x, x), 1)

 0.5983 -0.0341  2.4918  0.5983 -0.0341  2.4918  0.5983 -0.0341  2.4918
 1.5981 -0.5265 -0.8735  1.5981 -0.5265 -0.8735  1.5981 -0.5265 -0.8735
[torch.FloatTensor of size 2x9]
```

3. ### torch.ceil
   
   > torch.ceil(input, out=None) → Tensor
   
   天井函数，对输入`input`张量每个元素向上取整, 即取不小于每个元素的最小整数，并返回结果到输出。
   
   参数：
   
   - input (Tensor) – 输入张量
   - out (Tensor, optional) – 输出张量
   
   例子：
   
   ```python
   >>> a = torch.randn(4)
   >>> a
   
    1.3869
    0.3912
   -0.8634
   -0.5468
   [torch.FloatTensor of size 4]
   
   >>> torch.ceil(a)
   
    2
    1
   -0
   -0
   [torch.FloatTensor of size 4]
   ```
