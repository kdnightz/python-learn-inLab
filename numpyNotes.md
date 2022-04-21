# numpy [...]语法

```python
import numpy
>>> a = numpy.array([[1,2,3,4,5],[6,7,8,9,10],[1,2,3,4,5],[6,7,8,9,10]])
>>> a
array([[ 1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10],
       [ 1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10]])
>>> a[...,2] #表示遍历每行，2表示索引为2的所在列
array([3, 8, 3, 8])
>>> a[...,:2]#表示遍历每行，:2表示索引为<2的0，1所在的列
array([[1, 2],
       [6, 7],
       [1, 2],
       [6, 7]])
>>> a[...,::2]#表示遍历每行，2表示步长，选取多索引为0，2，4所在的列
array([[ 1,  3,  5],
       [ 6,  8, 10],
       [ 1,  3,  5],
       [ 6,  8, 10]])
a[None,...]#相当于插入维度，也想当于reshape(a,[1,4,4])
array([[[[ 1,  2,  3,  4,  5],
         [ 6,  7,  8,  9, 10],
         [ 1,  2,  3,  4,  5],
         [ 6,  7,  8,  9, 10]]]])
```



先随机产生一个[3,4,5]的[numpy](https://so.csdn.net/so/search?q=numpy&spm=1001.2101.3001.7020)数组。则该x维度是3，shape是（3,4,5），总共包含60个元素。

![](https://img-blog.csdnimg.cn/20190413165816790.png)

x[:,:,0] 意思是对[数组](https://so.csdn.net/so/search?q=%E6%95%B0%E7%BB%84&spm=1001.2101.3001.7020)x切片，可以想象成一个正方体数据，每次切下一个面的数据。第二维取0则得出来[3,4]大小的数组，即

![](https://img-blog.csdnimg.cn/2019041317165764.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzU1MjgxNg==,size_16,color_FFFFFF,t_70)

可以验证：

![](https://img-blog.csdnimg.cn/20190413171737581.png)

## 

**那么**[...,0]代表了什么？

首先...只能出现一次，就是说你可以，**[ : , : , : ],但是[ ... , ...]就会报错。**

使用了 ... 之后，数字0不再是元素的index 了 , 而是 轴（axis）。下面通过numpy.amax()（选出轴最大的元素）来具体说明。

![](https://img-blog.csdnimg.cn/20190413173501984.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzU1MjgxNg==,size_16,color_FFFFFF,t_70)

x大小为（3,4,5）

首先看axis=2，第一个数字189是从x[ ][ ] **[0]** 到 x[ ][ ] **[4]** 比较而得，因此一共有3*4=12元素

axis=1，第一个数字99是从x[ ]**[0]** [ ] 到 x[ ]**[3]** [ ] 比较而得，因此一共有3*5=15元素

同理，axis=0，第二个数字189是从x**[0]** [ ] [ ] 到 x**[2]** [ ] [ ] 比较而得，因此一共有4*5=20元素

axis=0时 比较的示意图：

![](https://img-blog.csdnimg.cn/20190413175522903.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzU1MjgxNg==,size_16,color_FFFFFF,t_70)

最直观的：函数所选的axis的值，就表明 x[ ][ ][ ] 的第几个方块号，**从0开始，代表第一个[ ]，**即x**[ ]** [ ] [ ]，所以维度与axis的对应关系为，对于维度为（3,4,5）的数组，axis=0 长度为3，axis=1长度为4，axis=2长度为5。

则[...,0]表示，与[:,:,0]等价：

![](https://img-blog.csdnimg.cn/20190413182818254.png)![](https://img-blog.csdnimg.cn/20190413183656890.png)

同时，还可以这样用。

![](https://img-blog.csdnimg.cn/20190413183115365.png)
