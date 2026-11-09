# 07 训练 RTC 与 chunk 对齐

需要查函数或类时，打开 [API 速查](Torch与Python_API速查.md)，按顶部题号表阅读。可以看着 API 写第一遍。

实现：[ex07_rtc.py](../exercises/ex07_rtc.py)。检查：`uv run python run.py 07`。建议 45—60 分钟。

## 两种时间先分清

- 控制步 h：动作序列中的第几个动作，或机器人执行的绝对步编号。
- 去噪时间 τ：Flow Matching 从噪声到动作的积分变量。

训练时延迟 d 是每个样本独立采样的整数，默认均匀取 0—6，包含 6；不是固定整批一个延迟。

## 要实现什么

1. sample_delays：返回 CPU long[B]；支持传入随机 generator。
2. make_rtc_batch：动作/噪声 [B,H,A]，τ[B]，delay[B]。前 d 步用干净动作，token_time=0；其余动作照常加噪，token_time=τ。返回 x、ε−a、token_time[B,H]、prefix_mask[B,H]。训练要求 0≤d<H。
3. rtc_loss：仅监督后缀；**每个样本按自己的后缀步数×A 平均，然后 batch 平均**。任一样本没有后缀时抛 ValueError。
4. rtc_sample：从 noise 开始 Euler 采样。每次调用模型前将已知前缀固定成 known_actions；前缀时间为 0，前缀速度更新强制为 0。返回时前缀也必须保持原值。不要改动输入，不建 autograd 图。推理测试允许 d=H。
5. next_chunk_offset：模型以 observation_step 为新 chunk 第 0 步，真正切换时为 switch_step，返回应该跳过多少步。负 offset 或 offset≥new_horizon 抛 ValueError。

对齐例子：旧 chunk 的第 20 步拿观测启动推理，若第 50 步才切换，新 chunk 从索引 **30** 开始；若第 24 步立即切换，则从索引 **4** 开始。启动位置、剩余旧动作、实际推理耗时是不同量。

## 分层提示

<details><summary>提示一：把所有样本一次广播出来</summary>

arange(H) 与 delay[:,None] 比较得到 prefix mask。后缀时间可由 τ[:,None] 与 ~prefix 组合。

先按普通 flow 构造，再覆盖已知前缀；target 仍可保留 ε−a，因为前缀不进入 loss。
</details>

<details><summary>提示二：为何需要每步固定</summary>

只在采样开始覆盖一次不够：模型若在前缀输出非零速度，下一步前缀会漂移。测试故意给前缀巨大速度，看你的采样是否保护了已知动作。

训练分母不能统一使用 H；否则延迟长的样本因为监督步少而被无意降权。
</details>

## 对照真实项目

当前 PiStar 主线不包含这轮 RTC 改动。可只读历史提交，不切换分支：

```powershell
git -C '..\openpi' show 'a6fb66c:src/openpi/models/pi0.py'
git -C '..\openpi' show 'a6fb66c:src/openpi/models/gemma.py'
git -C '..\openpi' show 'a6fb66c:src/openpi/policies/policy.py'
```

本题是训练过前缀条件的 RTC 路线。另一种推理时 VJP 引导路线并不等同于此题；复述项目时分别说明 `x−τv` 的动作估计、VJP 校正与已知前缀条件。

## 口头追问

- 为什么前缀不参与训练 loss，模型还能利用它？
- 为什么需要逐 token 时间，而不是给整条 chunk 一个 τ？
- 延迟超出训练覆盖范围会怎样？结果到达太晚、chunk 已耗尽时怎样处理？
- 你没有量化连续性就如实说“定性观察”；若补实验，可统计切换处动作跳变量和速度/加速度变化，再报告成功率与延迟。
