# 03 Transformer Block 与 adaRMS

需要查函数或类时，打开 [API 速查](Torch与Python_API速查.md)，按顶部题号表阅读。可以看着 API 写第一遍。

实现：[ex03_transformer.py](../exercises/ex03_transformer.py)。检查：`uv run python run.py 03`。建议 60—90 分钟。依赖 02。

## 四个模块

**RMSNorm**：沿最后特征维计算 x/√(mean(x²)+eps)，再乘可学习 weight；不减均值。平方与均值用 float32，结果回到输入 dtype。本题 weight 初始化为 1；本地 JAX Gemma 的等价参数形式是初值 0 的 scale，再乘 (1+scale)。

**GatedMLP**：down_proj(gelu(gate_proj(x), approximate="tanh") × up_proj(x))。本题对齐 Gemma 的 GELU 门控，不要因名字相似就写成 SiLU 的 SwiGLU。

**PreNorm block**：

- 先规范化，再 attention，再与原输入加残差。
- 对刚得到的结果规范化，再 MLP，再加第二次残差。
- 保持 [B,N,D]。attention 接收 allowed mask。

**AdaptiveRMSNorm**：

- x:[B,N,D]，cond 可以是 [B,C] 或 [B,N,C]。
- cond 经已提供的 Linear(C,3D)，顺序拆出 scale、shift、gate。
- 返回 RMS(x)×(1+scale)+shift，以及 gate。
- 样本级条件广播到全部 token；逐 token 条件分别作用。
- gate 不过 sigmoid，本函数不做残差。调制层已初始化为零，因此初始 scale/shift/gate 都为 0。

## 分层提示

<details><summary>提示一：残差连接不要写串</summary>

第二个 norm 的输入是 attention 残差相加后的结果。将两个步骤分别赋值，避免误用最初的 x。

cond 为 [B,C] 时，经过 Linear 得到 [B,3D]，需要补上 token 维；不能依赖偶然相同的 B 和 N 广播。
</details>

<details><summary>提示二：三个“门”不是一个东西</summary>

MLP 的 gate_proj 是特征相乘分支；adaRMS 的 gate 是控制残差分支的条件信号；attention mask 是允许关注的布尔关系。把三个名字分清。

半精度下先 square 再 float 已经可能溢出；先转 float32 再平方。
</details>

## 口头追问

- RMSNorm 与 LayerNorm 的差别是什么？为什么能使用不同长度序列？
- PreNorm 中残差路径对梯度有什么作用？
- adaRMS 的时间条件改变了什么？是否新增了一个动作 token？
- 为什么 gate=0 不是 sigmoid(0)=0.5？零初始化时哪条参数路径第一步可能没有梯度？

对照 [gemma.py](../../openpi/src/openpi/models/gemma.py) 的 RMSNorm、FeedForward 和 Block。不要把本题结构简化当成模型 checkpoint 的精确参数布局。
