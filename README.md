# VLA 二面手撕练习

已按确认的方案生成：**1 组张量热身 + 6 道核心题 + 2 道项目专项题**。题面、TODO 骨架、分层提示和 58 项测试都已提供，未放完整答案。

先从 [00 热身](notes/00_热身.md) 开始，再做 [01 状态输入](notes/01_状态输入.md)。今天的目标是能把“关节状态如何进入 π0.5”从原始数据讲到 Transformer，并写出 Attention。

## API 速查（可以边查边写）

打开 [Torch、NumPy 与 Python API 速查](notes/Torch与Python_API速查.md)。每项有用途、参数、形状变化和小例子，顶部按题号列出需要阅读的章节。做 00 时先看第 1—3 节即可。

[api_playground.py](api_playground.py) 用于随手运行例子，不影响练习文件。速查表中的 34 个 Python 示例块已在当前环境执行检查。

## 使用方法

在 PowerShell 中运行，不需要手动激活环境：

```powershell
Set-Location -LiteralPath 'E:\lzj\面试相关\手撕练习'
& '.\.venv\Scripts\python.exe' check_env.py
& '.\.venv\Scripts\python.exe' run.py 00
```

修改 `exercises/ex00_tensors.py` 的 TODO 后，再执行最后一行。也可以使用 `uv run python run.py 00`。

- 换题：`run.py 01`，编号为 00—08。
- 只测一个知识点：`run.py 02 -k all_masked`。
- 全部检查：`run.py all`。
- 默认遇到首个错误就停；直接调用 `python -m pytest -q` 可查看全部结果。
- 初始状态出现 **NotImplementedError 是正常的**：它表示尚未完成对应 TODO。后面的综合题会复用前面的实现。
- 测试是公开的行为契约；可以读断言，不要改断言来绕过错误。

## 题目顺序

| 编号 | 内容 | 首轮建议用时 | 本地代码关联 |
|---|---|---|---|
| [00](notes/00_热身.md) | reshape、transpose、广播、有效平均 | 20—30 分钟 | 所有模型张量操作 |
| [01](notes/01_状态输入.md) | 归一化、状态分桶、文本与 token mask | 35—45 分钟 | transforms / tokenizer |
| [02](notes/02_Attention.md) | 手写多头注意力、padding / causal mask | 60—90 分钟 | gemma Attention |
| [03](notes/03_Transformer.md) | RMSNorm、gated MLP、残差、adaRMS | 60—90 分钟 | gemma Block / RMSNorm |
| [04](notes/04_Flow_CFG.md) | Flow Matching、Euler、CFG | 45—60 分钟 | pi0 loss / sampling |
| [05](notes/05_动作专家.md) | 小型动作专家接通输入、时间和 mask | 60—90 分钟 | pi0 / gemma_pytorch |
| [06](notes/06_后训练.md) | 冻结、训练 step、梯度排查 | 35—50 分钟 | train_step |
| [07](notes/07_RTC.md) | 训练 RTC、前缀固定、chunk 对齐 | 45—60 分钟 | RTC 历史提交 |
| [08](notes/08_Value_Advantage.md) | two-hot、价值期望、N 步 adv、分位数 | 45—60 分钟 | 项目一价值与数据标注 |

依赖：00 → 02 → 03 → 05；04 为 05 的训练闭环服务。01、06、07、08 可单独练。00 是热身，不计入“6+2”。

## 四天路线

详见 [π0.5 代码细读与四天安排](notes/pi05_四天代码路线.md)。每天建议 5—6 小时，末尾预留休息和复述时间。优先读完整的数据流，再深入模块。

每题按以下顺序做：

1. 可以参考 API 速查，先写输入输出形状和一句算法描述。
2. 补 TODO；卡住 10 分钟再展开提示一，仍卡再看提示二。
3. 测试通过后，关掉实现，限时重写核心函数。
4. 回答题面最后的口头追问；测试通过不等于能解释。

完整答案暂不提供。之后可以针对你写的代码做 review，或按题号做模拟面试。

## 环境和范围

Python 3.11，CPU PyTorch、NumPy、pytest、matplotlib、einops；版本固定在 `uv.lock`。环境自检覆盖前向、反向、参数更新和 NumPy 桥接。

VS Code 解释器选：

```text
E:\lzj\面试相关\手撕练习\.venv\Scripts\python.exe
```

单独打开本目录时，`.vscode/settings.json` 已设置解释器。重建环境：`uv sync --frozen --link-mode copy`。

练习只运行小张量和小模型，无需下载预训练权重，也不依赖完整 JAX/OpenPI 环境。第 05 题明确采用教学简化；真实模型结构以相邻 `openpi` 仓库为准。岗位调研和原始选题理由保存在 [前期方案](notes/二面准备方案_待确认.md)，本 README 为确认后的执行入口。
