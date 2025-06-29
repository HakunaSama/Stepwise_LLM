我们经常看到register parameter和register buffer这样的函数调用，我们详细说一下这两个函数的用法：

register parameter：

```python
import torch
from torch import nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(3,3)
        #手动创建的可训练参数 (Parameter 会自动加入 parameters() / state_dict())
        self.parameter1=nn.Parameter(torch.randn(3))
        #显式注册一个参数到模块 (名字 → "parameter2")
        self.register_parameter("parameter2",nn.Parameter(torch.randn(3)))
        
    def forward(self,x):
        return self.linear(x)+self.parameter1+self.parameter2
    
if __name__=="__main__":
    model=TestModel()
    # ----------------------------------------
    # 打印所有可训练参数 (含名字与张量本身)
    # 这里会列出:
    #   linear.weight
    #   linear.bias
    #   parameter1
    #   parameter2
    # ----------------------------------------
    for n,p in model.named_parameters():
        print(n,p)
        
    print("*"*40)
    # ----------------------------------------
    # state_dict() 既包含上述参数，也包含子模块的缓冲区 (如 BatchNorm 的 running_mean)
    # 本例只有参数，没有缓冲 (buffers)
    # ----------------------------------------
    for n,s in model.state_dict().items():
        print(n,s)
        
    model.to("cuda")
    print("*"*40)
    # ----------------------------------------
    # 将整个模型搬到 GPU
    # 所有参数 / buffer 的 device 都会变为 cuda:0
    # ----------------------------------------
    for n,s in model.state_dict().items():
        print(n,s)
```

因为我们在声明Parameter类型的成员变量时，后台会自动调用register_parameter方法，所以上述两种写法本质上是一样的。在register_parameter这个方法内部，会做几件事，1.把parameter放到module parameters中，这样当我们把module parameters传递给优化器时，就自动包含了我们自己定义的这个parameter，这个parameter和我们定义的线性层的weight和bias是一样的，另外我们检查模型的state dict，可以发现parameter1和parameter2都在模型的dict里了，和我们定义的weight和bias是一样的，最后我们看一下把模型加载到gpu上，可以看到定义的线性层权重和parameter1和parameter2都被转移到了cuda上

结果：

```
### named_parameters() 初始状态 (CPU) ###
linear.weight    → torch.Size([3, 3]), device=cpu
linear.bias      → torch.Size([3]), device=cpu
parameter1       → torch.Size([3]), device=cpu
parameter2       → torch.Size([3]), device=cpu
****************************************
### state_dict() 初始状态 (CPU) ###
linear.weight    → torch.Size([3, 3]), device=cpu
linear.bias      → torch.Size([3]), device=cpu
parameter1       → torch.Size([3]), device=cpu
parameter2       → torch.Size([3]), device=cpu
****************************************
### state_dict() 迁移到 CUDA 后 ###
linear.weight    → torch.Size([3, 3]), device=cuda:0
linear.bias      → torch.Size([3]), device=cuda:0
parameter1       → torch.Size([3]), device=cuda:0
parameter2       → torch.Size([3]), device=cuda:0
```

register_buffer：

```python
import torch
from torch import nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(3,3)
        self.buffer1=torch.randn(3)
        self.register_buffer("buffer2",torch.randn(3))
        
    def forward(self,x):
        self.buffer1 +=1
        self.buffer2 +=1
        return self.linear(x)+self.parameter1+self.buffer1+self.buffer2
    
if __name__=="__main__":
    model=TestModel()
	print("*"*20,"parameters","*"*20)
    for n,p in model.named_parameters():
        print(n,p)
        
    print("*"*20,"state_dict","*"*20)
    for n,s in model.state_dict().items():
        print(n,s)
        
    model.to("cuda")
    print("*"*20,"to_cuda","*"*20)
    for n,s in model.state_dict().items():
        print(n,s)
```

register_buffer方法的作用时注册一个不需要训练的变量，这个方法需要我们手动来调用，有时候我们需要模型里定义一些不需要训练的常量时，比如position encoding里面的位置编码，或者一些不需要训练的变量，比如在batch norm里移动平均和移动方差的值，都需要调用register_buffer。

register_buffer和我们直接定义一个tensor成员变量有什么不同呢？在上面的代码中我们定义了一个类的tensor类型成员变量buffer1，同时注册了一个buffer2，在forward函数中我们使用了这两个buffer，然后我们检查他们是否会存在模型的parameters列表中，接着检查它们是否出现在模型的state_dict里，最后我们检查一下，他们是否会随着模型一起从cpu转移到gpu中，可以看到buffer1和buffer2都没有被保存到模型的parameters列表里，这意味着我们把parameters列表传递给优化器时，它们都不在里面不会被训练，

我们定义的类成员变量buffer1不在model的state_dict里，而我们注册的buffer2在model的state_dict里，也就是说我们注册到buffer里的变量会随着我们的模型一起被保存，将来可以方便加载，就像batch norm里面的均值和方差就是通过buffer来做到保存和加载的。

在模型转到gpu上时，同样，类的成员变量buffer1没有被转移到gpu上，而我们注册的buffer2被转移到了gpu上，

```
******************parameters**********************
linear.weight Parameters containing:
tensor([[-0.1417, -0.5482, -0.0386],
        [-0.1457,  0.5349,  0.1572],
        [-0.1665, -0.3184,  0.1730]], device='cpu', requires_grad=True)
linear.bias Parameters containing:
tensor([-0.2515,  0.4021,  0.5129], device='cpu', requires_grad=True)
******************state_dict**********************
linear.weight tensor([[-0.1417, -0.5482, -0.0386],
                      [-0.1457,  0.5349,  0.1572],
                      [-0.1665, -0.3184,  0.1730]], device='cpu')
linear.bias tensor([-0.2515,  0.4021,  0.5129], device='cpu')
buffer2 tensor([-1.0490,  0.0146,  0.7029], device='cpu')
******************to_cuda**********************
linear.weight tensor([[-0.1417, -0.5482, -0.0386],
                            [-0.1457,  0.5349,  0.1572],
                            [-0.1665, -0.3184,  0.1730]], device='cuda:0')
linear.bias tensor([-0.2515,  0.4021,  0.5129], device='cuda:0')
buffer2 tensor([-1.0490,  0.0146,  0.7029], device='cuda:0')
```

总结一下：

|                          | register_parameter | register_buffer |
| ------------------------ | ------------------ | --------------- |
| 加入模型的Parameters列表 | 会                 | 不会            |
| 加入模型的state_dict     | 会                 | 会              |
| 随着模型一起进行设备转移 | 会                 | 会              |

