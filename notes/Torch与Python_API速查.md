# 手撕练习用到的 Torch、NumPy 与 Python API

这份资料可以开在题目旁边查。第一遍允许查函数和参数；先练会把思路写成代码，第二遍再尝试不查。

**现在做 00，只读第 1—3 节就够。** 后面的 API 做到对应题再看，不需要一次背完。示例是函数用法，不是整题答案。

## 按题号查

| 题号 | 主要查哪些内容 |
|---|---|
| 00 张量热身 | 第 1—3 节：shape、reshape、transpose、广播、sum、clamp_min |
| 01 状态输入 | 第 9 节：NumPy 数组、linspace、digitize、字符串、padding |
| 02 Attention | 第 2、5、6 节：转置、矩阵乘法、mask、softmax、Linear |
| 03 Transformer | 第 3、6 节：平方均值、rsqrt、Module、Parameter、GELU、chunk |
| 04 Flow/CFG | 第 3、4、7、8 节：广播、噪声、mean、no_grad、循环 |
| 05 动作专家 | 第 2、5、6、8 节：cat、mask、ModuleList、时间编码 |
| 06 后训练 | 第 6—7 节：模型参数、requires_grad、backward、优化器 |
| 07 RTC | 第 3—5、8 节：arange、广播比较、where、randint |
| 08 Value/Advantage | 第 5、9—10 节：log_softmax、scatter_add_、percentile |

常用导入放在 Python 文件顶部：

```python
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
```

- torch：张量计算。
- nn：Linear、Module 等神经网络类。
- F：gelu、mse_loss 等函数；F 是我们给 functional 起的简写。
- np：NumPy，当前题目里主要处理数据和标签。
- math：Python 数学库，用于标量运算。

想试一下示例，可以复制到 [api_playground.py](../api_playground.py) 运行。代码块按节独立阅读；先运行上面的导入。文件中 `print(...)` 的结果会显示在终端。

## 1. 先认识张量、shape 和 dim

Tensor 可以先理解成“多维数组”。神经网络中的输入、输出、权重、梯度通常都是 Tensor。

```python
x = torch.randn(2, 3, 4)
print(x.shape)       # torch.Size([2, 3, 4])
print(x.ndim)        # 3：一共有三条轴
print(x.shape[0])    # 2：第 0 维长度
print(x.shape[-1])   # 4：最后一维长度
print(x.size(1))     # 3：与 x.shape[1] 相同
print(x.numel())     # 24：元素总数
B, N, D = x.shape    # 将三个长度分别赋给 B、N、D
```

对于 [B,N,D]：

| dim | 代表什么 | 例如 [2,3,4] |
|---|---|---|
| 0 | batch，样本 | 2 个样本 |
| 1 | token/序列位置 | 每个样本 3 个 token |
| 2 或 -1 | 特征 | 每个 token 4 个数 |

**dim 是轴编号，不是这条轴的长度。** `dim=1` 表示第 1 轴，不是“变成一维”。

常见符号：B=batch；N/P=token 数；D=特征宽度；h=注意力头数；H=动作步数；A=动作维数。实际变量名由函数决定。

### 属性、方法、函数怎么区分

| 写法 | 意思 |
|---|---|
| `x.shape` | 读取属性，不加括号 |
| `x.reshape(2, 12)` | 调用张量的方法 |
| `torch.reshape(x, (2, 12))` | 同一操作的函数式写法 |
| `x.sum(dim=1)` | 参数 dim 指定沿哪条轴求和 |
| `nn.Linear(4, 8)` | 创建一个层对象 |
| `layer(x)` | 把 x 送入已经创建的层 |

题目中的 `x: torch.Tensor`、`-> torch.Tensor` 是类型提示，不会自动转换数据。

## 2. 改形状：reshape、transpose、unsqueeze

### reshape：重新分组元素

`x.reshape(新形状)`：总元素数不变，按元素的逻辑顺序重新分组。只允许一个位置使用 -1，让 PyTorch 自动计算该长度。

```python
x = torch.arange(24)       # 0 到 23，形状 [24]
y = x.reshape(2, 3, 4)    # 形状 [2,3,4]
z = y.reshape(2, -1)      # 形状 [2,12]，-1 自动推得 12
print(y[0, 0])           # tensor([0, 1, 2, 3])
print(z.shape)           # torch.Size([2, 12])
```

常见错误：24 个数不能 reshape 成 [5,5]。

### transpose：交换两条轴

`x.transpose(dim0, dim1)`：只交换指定的两条轴，不是任意重新分组。

```python
x = torch.arange(24).reshape(2, 3, 4)
y = x.transpose(1, 2)
print(y.shape)           # [2,4,3]
print(x[0, 1, 2])        # 6
print(y[0, 2, 1])        # 仍是 6：这个元素的后两个索引交换了
```

`x.transpose(-2,-1)` 常用来转置最后两维，前面的 batch/head 维保持原位。

**reshape 成相同形状不代表与 transpose 内容相同。** 做多头注意力时，要先弄清楚“拆一条轴”和“交换轴”各发生在哪一步。

### view、contiguous：做题先掌握这个区别

| 方法 | 用途 |
|---|---|
| `x.view(...)` | 按新形状查看共享存储的数据，对 stride 有要求 |
| `x.reshape(...)` | 能共享存储时共享，必要时会复制，使用更方便 |
| `x.is_contiguous()` | 当前布局是否连续 |(连续是指内存中的顺序和看起来的顺序是不是一样的)
| `x.contiguous()` | 已连续则可直接返回，否则复制成连续布局 |

transpose 后通常不连续。合头时可以 reshape，或 contiguous 后 view。不要假设 reshape 永远不复制。

```python
x = torch.randn(2, 3, 4)
y = x.transpose(1, 2)
print(y.is_contiguous())     # False
z = y.contiguous().view(2, 12) # contiguous()是将内存中的顺序按照当前逻辑顺序改变，view()是按照指定的形状查看此张量的内存中的顺序
print(z.shape)               # [2,12]
```

### unsqueeze / squeeze：增加或移除长度为 1 的轴

```python
x = torch.randn(2, 3)
print(x.unsqueeze(-1).shape)  # [2,3,1]
print(x[:, :, None].shape)    # [2,3,1]，同一种效果
print(x.unsqueeze(1).shape)   # [2,1,3]
print(x[:, None, :].shape)    # [2,1,3]
print(x.unsqueeze(-1).squeeze(-1).shape)  # [2,3]
```

尽量指定 squeeze 的轴：直接 `squeeze()` 会移除所有长度为 1 的轴，B=1 时可能误删 batch。

### 其他形状操作：后面用到再查

| API | 示例及效果 |
|---|---|
| `x.permute(0,2,1)` | 同时指定所有轴的新顺序；[B,N,D]→[B,D,N] |
| `torch.cat([a,b], dim=1)` | 沿已有轴接长；[B,2,D]+[B,3,D]→[B,5,D] |
| `torch.stack([a,b], dim=1)` | 增加一条轴；两个 [B,D]→[B,2,D] |
| `x.chunk(3, dim=-1)` | 分块；本题最后维为 3D 时返回三个 [...,D] |
| `x.expand(B,N,D)` | 把长度为 1 的轴广播展开；其他轴不能随便改长度 |
| `x.expand_as(other)` | 按 other 的形状展开 |
| `x.repeat(1,3,1)` | 实际重复数据；参数是每条轴的重复次数 |

expand 可能让多个位置共享同一块存储，不要直接原地写入展开后的张量。chunk 在长度不能按要求拆分时有额外边界行为；本轮 scale/shift/gate 用的是整除情形。

## 3. 广播、求和、平均：00 题最常用

### 广播怎么判断

从右向左比较两者形状：对应长度相同，或其中一个是 1，或一侧缺少这条轴，就可以对齐扩展。广播是计算规则，不需要手写循环复制数据。[官方广播说明](https://docs.pytorch.org/docs/2.14/notes/broadcasting.html)
广播用于*形状不同时帮助它对齐的规则，先把两个shape块全部推到最右边，shape长度相同或者其中一个为1就可以计算，长度为1时操作就是同行或者列补全同一个数后再逐个相乘

```text
[2,3,4]
[2,3,1]   可以：同一个 token 的系数应用到 4 个特征

[2,3,4]
[2,3]     不可以：从最右侧比较，4 与 3 不兼容

[2,3,4]
[  3,1]   可以：同一套 [3,1] 系数用于两个样本
```

```python
x = torch.ones(2, 3, 4)
coefficient = torch.tensor([[1., 2., 3.], [4., 5., 6.]]) #注意着玩意是shape是[2,3]（看shape的方法是从里到外因此对应的广播对其方式是最右边对其）,而[[[1., 2., 3.]], [[4., 5., 6.]]]这玩意才是[2,1,3] 

y = x * coefficient.unsqueeze(-1)  # *代表逐个元素相乘，
print(y.shape)       # [2,3,4]
print(y[0, 1])       # [2.,2.,2.,2.]
```

注意：有些错误形状碰巧也能广播，程序不报错却计算错了。要写出每条轴的含义。

### sum / mean：沿某条轴做归约

```python
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
print(x.sum(dim=0))                  # [5.,7.,9.]，沿行方向把两行加起来 （行间之和）
print(x.sum(dim=1))                  # [6.,15.]，每行内部相加 （列间之和）
print(x.sum(dim=1, keepdim=True))    # [[6.],[15.]]，形状 [2,1] 
print(x.mean(dim=1))                 # [2.,5.]，每行内部平均
print(x.mean())                      # 3.5，所有元素一起平均
```

`keepdim=True`：被归约的轴保留，长度变成 1，方便后续广播。

对于 [B,N,D]：

| 写法 | 输出形状 |
|---|---|
| `x.sum(dim=1)` | [B,D] | #被和归约消失
| `x.sum(dim=1, keepdim=True)` | [B,1,D] | #保留被归约维度，方便广播
| `x.mean(dim=-1, keepdim=True)` | [B,N,1] |
| `x.sum(dim=(1,2))` | [B] |
| `x.mean()` | 标量 Tensor，形状 [] |

### bool mask 乘法与有效数

```python
valid = torch.tensor([[True, False, True], [False, True, False]])
print(valid.sum(dim=1))      # [2,1]；True 当成 1，False 当成 0
x = torch.ones(2, 3, 4)
masked = x * valid.unsqueeze(-1)  #相当于给每一个维度套多了一个括号[[[True], [False], [True]], [[False], [True], [False]]]
print(masked[0, 1])          # [0.,0.,0.,0.]
```

这里演示有限数值上的乘法 mask。如果输入本身含 NaN，NaN×0 仍为 NaN，不能靠乘 0 清除。

### clamp_min：给分母设置下限

```python
count = torch.tensor([0., 2., 5.])
print(count.clamp_min(1))    # [1.,2.,5.]；小于 1 的换成 1
print(count.clamp(1, 3))     # [1.,2.,3.]；同时限制上下界
```

`clamp_min` 只处理分母过小的问题；有效平均的分子和分母究竟沿哪一维计算，仍要按题目写。

### 数学函数

| API | 意思 | 用在哪 |
|---|---|---|
| `x.square()` 或 `x ** 2` | 逐元素平方 | MSE、RMS |
| `torch.sqrt(x)` 或 `x.sqrt()` | 平方根 | RMS |
| `torch.rsqrt(x)` | 1/√x | RMSNorm |
| `x.abs()` | 绝对值 | 看梯度大小 |
| `x.sin()`、`x.cos()` | 弧度制正弦、余弦 | 时间编码 |
| `x.exp()`、`x.log()` | 指数、自然对数 | 理解概率运算 |
| `math.sqrt(d)` | Python 标量开方 | 注意力缩放 |
| `math.pi` | π | 时间编码 |

## 4. 创建张量、dtype、device

| API | 示例/说明 |
|---|---|
| `torch.tensor(data)` | 从 Python 数据新建张量，如 `torch.tensor([1.,2.])` |
| `torch.zeros(2,3)` / `ones(2,3)` | 创建 [2,3] 的全 0 / 全 1 |
| `torch.full((2,3), 0.5)` | 创建指定形状、填充指定值；shape 用元组 |
| `torch.randn(2,3)` | 标准正态噪声 |
| `torch.rand(2,3)` | [0,1) 均匀随机数 |
| `torch.randn_like(x)` | 与 x 同形状、dtype、device 的正态噪声；x 应为浮点类型 |
| `torch.zeros_like(x)` / `ones_like(x)` | 默认继承 x 的形状、dtype、device |
| `torch.full_like(x, 0.5)` | 继承 x 属性并填值；x 为整数时 0.5 会被转成整数 |
| `x.new_zeros(2,5)` | 新形状，继承 x 的 dtype/device |
| `torch.arange(4)` | [0,1,2,3]，不包含右端点 |
| `torch.linspace(0,1,5)` | [0,.25,.5,.75,1]，包含两端 |
| `torch.randint(0,7,(B,))` | B 个整数，范围 0—6；上界 7 不包含 |
| `torch.manual_seed(7)` | 设置随机种子，便于复现 |
| `torch.Generator().manual_seed(7)` | 独立随机生成器，可传给 randint 等函数 |

`(B,)` 是只有一个元素的元组；逗号不可省，`(B)` 只是 B 本身。

```python
x = torch.tensor([1, 2, 3])
print(x.dtype)        # torch.int64
print(x.device)       # cpu
y = x.float()
print(y.dtype)        # torch.float32
mask = torch.tensor([True, False], dtype=torch.bool)
ids = torch.tensor([2, 3], dtype=torch.long)
noise = torch.randn(2, 3, dtype=y.dtype, device=y.device)
```

常用转换：

- `x.float()`：转 float32；RMSNorm 先转再平方，避免半精度平方溢出。
- `x.long()`：转 int64；常用于索引。
- `x.bool()`：转布尔。
- `x.to(dtype=..., device=...)`：显式转换；`x.to(other.dtype)` 只改 dtype。
- `x.cpu()`：移到 CPU。
- `x.detach().cpu().numpy()`：转 NumPy 做观察，已经切断梯度，不能再用来搭可微训练路径。

从已有 Tensor 复制时要区分：

| 写法 | 是否复制存储 | 是否保留梯度联系 |
|---|---|---|
| `y = x` | 否，只是同一个对象的别名 | 保留 |
| `x.clone()` | 是 | 保留 |
| `x.detach()` | 通常共享存储 | 切断 |
| `x.detach().clone()` | 是 | 切断 |

`torch.tensor(x)` 会新建并切断已有张量的梯度联系；不要用它随手“包装”中间结果。Tensor API 可查 [官方索引](https://docs.pytorch.org/docs/2.14/tensors.html)。

## 5. Attention 与 RTC 常用的 mask 操作

### 切片和布尔索引

```python
x = torch.arange(24).reshape(2, 3, 4)
print(x[:, 1, :].shape)      # [2,4]：取所有样本的第 1 个 token（单个1不保留维度，对比见下面保留维度）
print(x[:, 1:2, :].shape)    # [2,1,4]：切片保留维度 token 轴 （起始位置:结束位置，此时相当于拿出1）
print(x[:, -2:, :].shape)    # [2,2,4]：最后两个 token
print(x[..., 0].shape)       # [2,3]：... 表示前面所有轴
valid = torch.tensor([[True, False, True], [False, True, False]])
print(x[valid].shape)        # [3,4]：把三个有效 token 取出来
# 相当于取出valid中的True，也就是[0,0],[0,2],[1,1],也就是对应x中的

[[ 0,  1,  2,  3],   # 样本 0，token 0
[ 8,  9, 10, 11],    # 样本 0，token 2
[16, 17, 18, 19]]    # 样本 1，token 1
所以是[3,4]
```

布尔索引会压缩被选择的轴；需要保持 [B,N,D] 时通常用 where 或乘 mask。

### 比较、逻辑运算、any/all

| 写法 | 意思 |
|---|---|
| `x > 0`、`x >= y`、`x == 0` | 逐元素比较，返回 bool 张量 |
| `~mask` | 逐元素取反，mask 应为 bool |
| `mask_a & mask_b` | 逐元素“且” |
| `mask_a \| mask_b` | 逐元素“或” |
| `mask.any(dim=-1)` | 这一行是否至少一个 True |
| `mask.all(dim=-1)` | 这一行是否全为 True |
| `torch.isfinite(x)` | 逐元素判断是否为有限数，NaN/±Inf 返回 False |

多个元素的 mask 用 &、|、~，不要用 Python 的 and/or/not。比较组合要加括号：`(x > 0) & (x < 1)`。

```python
position = torch.arange(4)[None, :] #none代表此处放一个长度为1的维度  # [1,4]
# position = torch.arange(4) 就是[4]，那个1维度用来广播的
length = torch.tensor([1, 3])[:, None] # [2,1]
# 同length = torch.tensor([1, 3]).unsqueeze(-1)
valid = position < length             # 广播成 [2,4]
[[1,2,3,4],[1,2,3,4]]<[[1,1,1,1],[3,3,3,3]]
print(valid)
# [[ True, False, False, False],
#  [ True,  True,  True, False]]
```

### masked_fill 与 where

`x.masked_fill(mask,value)`：mask 为 True 的位置替换成 value。注意它的 True 表示“替换”，跟题目 allowed=True 表示“允许关注”不是同一个含义。

```python
scores = torch.tensor([[2., 4., 6.]])
allowed = torch.tensor([[True, False, True]])
masked_scores = scores.masked_fill(~allowed, float("-inf")) #~allowed代表取反bool
print(masked_scores)  # [[2., -inf, 6.]]
weights = masked_scores.softmax(dim=-1)
print(weights)       # 中间位置为 0，其余两项和为 1

softmax先对指定维度取指数[e², e⁻∞, e⁶] ≈ [7.389, 0, 403.429]然后除以和得到权重weights ≈ [0.018, 0, 0.982]


a = torch.tensor([10., 20., 30.])
b = torch.tensor([1., 2., 3.])
choose_a = torch.tensor([True, False, True])
print(torch.where(choose_a, a, b))    # [10.,2.,30.]
```

torch.where(condition,a,b)：True 取 a，False 取 b，三个输入按广播规则对齐。它不会让两侧表达式“只计算其中一个”，所以不能靠 where 事后掩盖除零或 NaN 的梯度问题。

整行都是 -inf 的 softmax 会产生 NaN；02 题要求你单独处理这种情况。

### 矩阵乘法 @ 与逐元素乘法 *

```python
a = torch.randn(2, 3)
b = torch.randn(3, 4)
print((a @ b).shape)       # [2,4]，矩阵乘法
print((a * a).shape)       # [2,3]，对应位置相乘
q = torch.randn(2, 3, 5, 4)
k = torch.randn(2, 3, 7, 4)
print((q @ k.transpose(-2, -1)).shape)   # [2,3,5,7]
```

`torch.matmul(a,b)` 与这里的 @ 等价；高维时前面的维度按 batch 广播，最后两维做矩阵乘法。

### softmax、log_softmax、tril

| API | 作用 |
|---|---|
| `scores.softmax(dim=-1)` | 沿最后轴归一化成概率，每行和为 1 |
| `F.softmax(scores,dim=-1)` | 函数式写法 |
| `scores.log_softmax(dim=-1)` | 直接稳定地计算 log 概率 |
| `F.log_softmax(scores,dim=-1)` | 同上 |
| `torch.ones(N,N,dtype=torch.bool).tril()` | 下三角含对角线为 True，用作因果关系 |

```python
print(torch.ones(3, 3, dtype=torch.bool).tril())
# [[True, False, False],
#  [True, True,  False],
#  [True, True,  True]]
```

## 6. 类和神经网络层：nn.Module 到底怎么用

### nn.Module、__init__、self、forward

先看一个与题目无关的小投影层：

```python
class SmallProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.proj(x)

model = SmallProjection(4, 6)
x = torch.randn(2, 3, 4)
y = model(x)
print(y.shape)                  # [2,3,6]
print(model.proj.weight.shape)  # [6,4]
```

逐行解释：

- `class ... (nn.Module)`：定义模型类，继承 PyTorch 的模块机制。
- `__init__`：创建模型对象时执行，定义需要长期保存的层和参数。
- `super().__init__()`：先初始化父类，才能正常注册子层和参数。
- `self`：当前这个模型对象。
- `self.proj`：模型保存的层，以后多次 forward 使用同一套参数。
- `forward`：定义输入如何计算成输出。
- `model(x)`：通过 Module 的调用机制执行 forward；正常使用这个写法。
- 不要在每次 forward 时重新创建 Linear，否则会不断新建参数。

模块的参数注册与调用机制见 [nn.Module 官方说明](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html)。

### nn.Linear：只变最后一维

`nn.Linear(in_features,out_features,bias=True)` 创建线性层。输入 [...,in_features] → 输出 [...,out_features]，前面的 batch/token 维保留。计算相当于 xWᵀ+b，weight 形状为 [out_features,in_features]。[Linear 官方说明](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html)

```python
layer = nn.Linear(4, 8, bias=False)
x = torch.randn(2, 5, 4)
print(layer(x).shape)    # [2,5,8]
print(layer.weight.shape) # [8,4]
print(layer.bias)       # None
```

创建层和调用层是两步：`layer = nn.Linear(...)`，再 `layer(x)`。它不是先把全部 token 展平后再投影。

### Parameter、Sequential、ModuleList

| 类/API | 意思 | 本轮用途 |
|---|---|---|
| `nn.Parameter(tensor)` | 把张量标记为可注册的模型参数，默认参与求导 | RMSNorm 的 weight |
| `nn.Sequential(layer1,layer2,...)` | 自动依次调用各层 | 小型 time_mlp |
| `nn.ModuleList([...])` | 注册一组模块；自己写 for 循环调用 | 多层 Transformer |
| `nn.init.zeros_(tensor)` | 将现有张量原地初始化为 0 | adaRMS 初始调制参数 |
| `nn.init.normal_(tensor,std=...)` | 正态分布初始化 | 理解测试中的参数设置 |

```python
network = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 2))
print(network(torch.randn(3, 4)).shape)  # [3,2]

layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
x = torch.randn(3, 4)
for layer in layers:
    x = layer(x)
print(x.shape)                          # [3,4]
```

普通 Python list 可以存对象，但其中的层不会像 ModuleList 一样自动注册到父模型。ModuleList 本身没有把层顺序执行的 forward。

### 激活函数与分块

| API | 作用 |
|---|---|
| `F.gelu(x, approximate="tanh")` | 本题 Gemma 风格 MLP 使用的 GELU 近似 |
| `F.silu(x)` / `nn.SiLU()(x)` | SiLU，x×sigmoid(x)，时间 MLP 使用 |
| `torch.tanh(x)` / `nn.Tanh()(x)` | 双曲正切 |
| `x.chunk(3,dim=-1)` | 把最后轴分三段，适合 scale/shift/gate |

```python
combined = torch.randn(2, 5, 12)
first, second, third = combined.chunk(3, dim=-1)
print(first.shape)   # [2,5,4]
```

本题 adaRMS 的 gate 直接使用，不需要因名字叫 gate 就加 sigmoid。

## 7. 梯度、冻结与优化器

### 一个最小求导例子

```python
x = torch.tensor([2., 3.], requires_grad=True)
loss = x.square().sum()
loss.backward()
print(x.grad)    # [4.,6.]，因为 d(x²)/dx=2x
print(loss.item())  # 13.0，取出 Python 数值用于记录
```

| API/属性 | 意思 |
|---|---|
| `tensor.requires_grad` | 当前张量是否需要梯度跟踪 |
| `p.requires_grad_(False)` | 就地修改参数的梯度开关 |
| `loss.backward()` | 从标量 loss 反向求导，默认累加到 grad |
| `p.grad` | 参数梯度；没有计算到时常为 None |
| `loss.item()` | 单元素 Tensor → Python 数字，会脱离计算图 |
| `model.parameters()` | 遍历模型参数 |
| `model.named_parameters()` | 遍历 (名称,参数)演示见下面 |
| `model.train()` / `model.eval()` | 设置训练/评估模式，不等于开关梯度 |
| `torch.no_grad()` | 作用域内不记录反向图，常用于采样 |
| `torch.nn.utils.clip_grad_norm_(parameters,max_norm)` | 原地裁剪总体梯度范数，发生在 backward 后 |

普通中间张量的 grad 默认不保留；参数这种叶子张量通常会有 grad。排查时先确定你查看的是哪一种张量。

```python
model = nn.Linear(4, 2)
for name, parameter in model.named_parameters():
    print(name, tuple(parameter.shape), parameter.requires_grad)
# weight (2,4) True  
# bias   (2,)  True
# 循环输出两次，说明每一个nn.Linear内部有两组可学习参数
# - weight：形状 [2,4]。每个输出需要 4 个权重，共有 2 个输出。
# - bias：形状 [2]。每个输出还有一个偏置。
# 此处展示一下这俩玩意

with torch.no_grad(): # 在torch.no_grad()下执行，作用域内不记录反向图，常用于采样（只看看效果，节省内存）
    result = model(torch.randn(3, 4))
print(result.requires_grad)  # False
```

no_grad 也可以写成函数上方的 `@torch.no_grad()` 装饰器，表示整个函数执行时不开反向图。

### 优化器相关 API

| 写法 | 意思 |
|---|---|
| `torch.optim.SGD(model.parameters(),lr=0.01)` | 创建 SGD 优化器，lr 是学习率 |
| `torch.optim.AdamW(model.parameters(),lr=1e-3)` | 创建 AdamW |
| `optimizer.zero_grad(set_to_none=True)` | 清理上次梯度，避免无意累加 |
| `optimizer.step()` | 用当前 grad 更新优化器管理的参数 |
| `F.mse_loss(pred,target)` | 默认对所有元素的平方误差取平均 |
| `F.mse_loss(pred,target,reduction="none")` | 保留逐元素误差，以便自己加 mask |

关键区别：冻结 backbone 的参数，仍可让梯度通过它的运算传到前面的 adapter；如果对整段 backbone forward 用 no_grad，可能把 adapter 的梯度也切断。

函数名末尾单个下划线常表示原地修改，如 `zero_`、`requires_grad_`、`clip_grad_norm_`。它与 Python 特殊方法两侧的双下划线 `__init__` 不是一回事。

一次训练主要做这几步
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

optimizer.zero_grad(set_to_none=True)
pred = model(x)
loss = F.mse_loss(pred, target)
loss.backward()
optimizer.step()
```

| 代码 | 做什么 |
|---|---|
| `zero_grad(...)` | 清理上一次留下的梯度 |
| `model(x)` | 计算当前预测 |
| `mse_loss(...)` | 衡量预测与正确答案的差距 |
| `loss.backward()` | 算出各参数对误差的影响，存到参数的 `.grad` 中 |
| `optimizer.step()` | 根据这些梯度更新参数 |

## 8. Flow 与 RTC：时间、循环和随机数

### 样本时间扩展到动作张量

```python
tau = torch.tensor([0.2, 0.8])  # [B] tau 表示去噪过程中的时间，也就是样本时间
print(tau[:, None, None].shape) # [B,1,1] 要与动作张量[B,H,A]对齐（*），所以扩展维度然后广播即可
print(tau[:, None].shape)       # [B,1]
```
B：样本数量。
H：每条样本预测多少步动作，这里不是注意力头数。
A：每步动作有多少个数，比如关节控制量。

将 [B] 直接乘 [B,H,A]，会从最后一维对齐，不会自动识别它是 batch 条件。先添加长度为 1 的轴。

### 循环及调用函数参数

初始噪声
   ↓ 模型预测如何变化，更新一次
中间结果
   ↓ 再预测、再更新
……
最终动作

```python
steps = 4
for i in range(steps): 
    print(i, 1 - i / steps)
# 0 1.0
# 1 0.75
# 2 0.5
# 3 0.25
# 依旧序号（方括号）-1（从零开始），数量（圆括号）就是数量

```
range 不包含右端点。上面只是列出模型调用时刻；状态更新要按题目自己实现。


`velocity_fn` 是“函数作为参数传进来”：

```python
def inspect_function(fn, x):
    return fn(x)

def double(x):
    return x * 2

print(inspect_function(double, torch.tensor([3.])))  # [6.]
```

传 `double` 是传函数本身；写 `double(x)` 是立即执行并传结果。

### 随机延迟、时间频率

```python
generator = torch.Generator().manual_seed(7)
delays = torch.randint(0, 7, (8,), generator=generator)
print(delays)   # 每个元素在 0—6，包含 6
'''创建随机数生成器，设定种子 7。
生成形状为 [8] 的随机整数张量。
每个数可以是 0、1、2、3、4、5、6，不包括 7。
'''

fraction = torch.linspace(0., 1., 4)
# fraction ≈ [0, 0.3333, 0.6667, 1] 分成四个数一共三段
period = 0.01 * (1.0 / 0.01) ** fraction
print(period)   # 从 0.01 到 1，按相同比例递增
```

额外读源码时会遇到 `torch.distributions.Beta(1.5,1.0).sample((B,))`：从 Beta 分布采样 B 个数。练习 04 已传入 tau，不要求你重新实现 Beta 采样。

## 9. NumPy 与 Python：状态文本、标签处理

### NumPy 基本函数

| API | 示例/用途 |
|---|---|
| `np.array([1.,2.])` | 创建数组 |
| `np.asarray(data,dtype=np.float32)` | 转成指定 dtype 的数组，适当情况下可以复用数据 |
| `np.zeros(5,dtype=np.int64)` | 创建长度 5 的整数 0 数组 |
| `np.ones_like(x,dtype=bool)` | 按 x 的形状创建全 True |
| `np.arange(5)` | [0,1,2,3,4] |
| `np.linspace(-1,1,5)` | [-1,-.5,0,.5,1] |
| `np.clip(x,low,high)` | 限制上下界；题 01 的归一化不要求 clip |
| `x.astype(np.int64)` | 转数据类型 |
| `x.sum(axis=1)` | 对第 1 轴求和 |
| `x.mean(axis=1,keepdims=True)` | 保留归约轴，注意 NumPy 是 keepdims |
| `x[mask]` | 取满足 bool mask 的元素 |
| `np.isfinite(x)` | 判断有限值 |
| `len(x)` | 第一条轴的长度；一维数组就是元素数 |

PyTorch 用 dim/keepdim，NumPy 常用 axis/keepdims，不要混写关键字。NumPy 没有 PyTorch 的自动求导图。

### digitize：查数值落在哪个区间

对于递增 bins，默认 right=False 时，返回 i，使 bins[i−1]≤x<bins[i]；小于首边界返回 0，达到或超过末边界返回 len(bins)。[digitize 官方说明](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html)

```python
edges = np.array([0., 10., 20.])
values = np.array([-2., 0., 7., 10., 20., 30.])
print(np.digitize(values, edges))  # [0,1,1,2,3,3]
```

这个返回值是区间位置。项目再做减 1，是项目选择的编号约定；不是 digitize 默认就返回词表 token。

### percentile：求分位数

```python
numbers = np.array([0., 10., 20., 30.])
print(np.percentile(numbers, 50))  # 15.0
print(np.percentile(numbers, 70))  # 21.0
```

q 使用 0—100，默认线性插值，阈值不必等于某个真实样本值。要先挑选参与统计的数据，再求阈值。[percentile 官方说明](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html)

### 字符串处理

```python
text = "  pick_up\nplug  "
cleaned = text.strip().replace("_", " ").replace("\n", " ")
print(cleaned)                         # pick up plug
numbers = [3, 15, 200]
joined = " ".join(map(str, numbers))
print(joined)                          # 3 15 200
print(f"Values: {joined}")              # Values: 3 15 200
```

| 写法 | 意思 |
|---|---|
| `text.strip()` | 删除开头、结尾的空白 |
| `text.replace(old,new)` | 替换字符串内容 |
| `str(value)` | 数字等对象转字符串 |
| `map(str,numbers)` | 对每个数执行 str |
| `" ".join(strings)` | 用空格连接字符串 |
| `f"Values: {joined}"` | 把变量值放进字符串 |
| `"\n"` | 字符串中的换行符，不是普通字母 n |

### padding 用得到的列表与切片

```python
items = [4, 8, 15]
print(items[:2])         # [4,8]，不含索引 2
print(items[-1])         # 15，最后一项
print(items + [0] * 2)   # [4,8,15,0,0]
ids = np.zeros(5, dtype=np.int64)
ids[:len(items)] = items
print(ids)               # [4,8,15,0,0]
```

`ids[:n]` 的目标长度必须与写入列表长度一致；超长情况先处理截断。

### 其他 Python 语法

| 写法 | 意思 |
|---|---|
| `a,b = function()` | 函数返回两个结果并分别接收 |
| `return a,b` | 返回二元组 |
| `for index,value in enumerate(items)` | 同时遍历下标和值 |
| `for a,b in zip(xs,ys)` | 成对遍历两个序列 |
| `min(a,b)`、`max(a,b)` | Python 标量取较小/大值 |
| `H // heads` | 整数除法 |
| `H % heads` | 取余，用于检查整除 |
| `if value is None:` | 判断没有提供值，不等于数值 0 |
| `raise ValueError("说明")` | 输入不合法时主动报错 |
| `assert condition` | 调试时确认条件，不成立则报 AssertionError |
| `{name: value}` | 创建字典，题 06 的梯度报告使用 |

## 10. two-hot 会用到的索引写入

### floor、clamp 和 long

```python
coordinate = torch.tensor([1.25, 3.8])
lower = coordinate.floor().long()
fraction = coordinate - lower
print(lower)      # [1,3]
print(fraction)   # [0.25,0.8]，浮点打印可能略有误差
```

- `floor()` 向下取整，输出仍可为浮点 dtype。
- `long()` 转成索引需要的整数 dtype。
- `clamp_max(n)` 限制索引最大值。
- 边界与权重计算按 08 题面处理，别让最大标签的权重消失。

### scatter_add_：按下标“加到”指定位置

以二维目标 dst、dim=1 为例：index[b,j] 指定目标列；将 src[b,j] 累加到 dst[b,index[b,j]]。重复下标会累加。[scatter_add_ 官方说明](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.scatter_add_.html)

```python
dst = torch.zeros(2, 4)
index = torch.tensor([[1, 1], [0, 3]], dtype=torch.long)
src = torch.tensor([[0.2, 0.8], [0.4, 0.6]])
dst.scatter_add_(dim=1, index=index, src=src)
print(dst)
# [[0.,1.,0.,0.],
#  [0.4,0.,0.,0.6]]
```

本轮用法保持 index 和 src 同形状，index 为 long、device 与目标一致。它不会替你随意广播；特别是 [B] 往 [B,K] 写入时，常需先变成 [B,1]。

如果更熟悉 one_hot，也可查 `F.one_hot(index,num_classes=K)`：输入整数索引 [B]，输出整数 [B,K]，做浮点加权前可转成标签的浮点 dtype。两种方式选一种理解即可。

## 11. 如何看报错与验证一个 API

| 报错/现象 | 优先检查 |
|---|---|
| size mismatch / sizes must match | 每条轴的长度，是否缺少 unsqueeze |
| mat1 and mat2 shapes cannot be multiplied | Linear 输入最后维；@ 的内维是否相同 |
| view size is not compatible | transpose 后 stride，不妨 reshape 或 contiguous |
| Boolean value of Tensor ... ambiguous | 对多元素 Tensor 用了 if/and/or；是否应 any/all |
| loss 或梯度 NaN | 全遮挡 softmax、除零、半精度溢出 |
| grad 是 None | requires_grad、detach/no_grad、是否连到 loss、是否叶子张量 |
| expected scalar type ... | float/long/bool 类型混用 |
| Expected all tensors ... same device | 创建 mask、时间时没有沿用输入 device |

调试时先打印三个东西：

```python
x = torch.randn(2, 3, 4)
print(x.shape, x.dtype, x.device)
```

比较两个结果：

```python
actual = torch.tensor([0.1 + 0.2])
expected = torch.tensor([0.3])
torch.testing.assert_close(actual, expected)   # 考虑浮点误差；失败会报差异
print(torch.allclose(actual, expected))       # 返回 Python bool
```

查本地帮助：`help(torch.reshape)`、`help(nn.Linear)`。想退出终端里的分页帮助可按 q。

**第一遍的目标：知道该查哪个函数，并能解释维度为什么这么变。先允许自己看着写。**
