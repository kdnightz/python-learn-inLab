1. **load_state_dict(state_dict)**
   
   将`state_dict`中的`parameters`和`buffers`复制到此`module`和它的后代中。`state_dict`中的`key`必须和 `model.state_dict()`返回的`key`一致。 `NOTE`：用来加载模型参数。
   
   参数说明:
   - state_dict (dict) – 保存`parameters`和`persistent buffers`的字典。
2. #### state_dict(destination=None, prefix='')[source]
   
   返回一个字典，保存着`module`的所有状态（`state`）。
   
   `parameters`和`persistent buffers`都会包含在字典中，字典的`key`就是`parameter`和`buffer`的 `names`。
3. ### class torch.nn.Sequential(* args)
   
   一个时序容器。`Modules` 会以他们传入的顺序被添加到容器中。当然，也可以传入一个`OrderedDict`。
   
   为了更容易的理解如何使用`Sequential`, 下面给出了一个例子:
   
   ```python
   # Example of using Sequential
   
   model = nn.Sequential(
             nn.Conv2d(1,20,5),
             nn.ReLU(),
             nn.Conv2d(20,64,5),
             nn.ReLU()
           )
   # Example of using Sequential with OrderedDict
   model = nn.Sequential(OrderedDict([
             ('conv1', nn.Conv2d(1,20,5)),
             ('relu1', nn.ReLU()),
             ('conv2', nn.Conv2d(20,64,5)),
             ('relu2', nn.ReLU())
           ]))
   ```



4. ### class torch.nn.CrossEntropyLoss(weight=None, size_average=True)
   
   此标准将`LogSoftMax`和`NLLLoss`集成到一个类中。
   
   当训练一个多类分类器的时候，这个方法是十分有用的。
   
   - weight(tensor): `1-D` tensor，`n`个元素，分别代表`n`类的权重，如果你的训练样本很不均衡的话，是非常有用的。默认值为None。
   
   - 调用时参数：
   
   - input : 包含每个类的得分，`2-D` tensor,`shape`为 `batch*n`
   
   - target: 大小为 `n` 的 `1—D` `tensor`，包含类别的索引(`0到 n-1`)。
     
     Loss可以表述为以下形式：
     
     ![](C:\pywork\learn\imgs\2022-04-16-11-50-52-image.png)
     
     当`weight`参数被指定的时候，`loss`的计算公式变为：
     
     ![](C:\pywork\learn\imgs\2022-04-16-11-51-13-image.png)
     
     计算出的`loss`对`mini-batch`的大小取了平均。
     
     形状(`shape`)：
     
     Input: (N,C) `C` 是类别的数量
   
   - Target: (N) `N`是`mini-batch`的大小，0 <= targets[i] <= C-1


