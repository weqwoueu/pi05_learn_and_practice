# 04 Flow Matching、Euler 与 CFG

需要查函数或类时，打开 [API 速查](Torch与Python_API速查.md)，按顶部题号表阅读。可以看着 API 写第一遍。

实现：[ex04_flow.py](../exercises/ex04_flow.py)。检查：`uv run python run.py 04`。建议 45—60 分钟。

## 统一时间方向

本题与仓库一致：τ=1 是噪声，τ=0 是动作。轨迹控制步用 H 表示，不要与去噪时间 τ 混淆。

给定真实动作 a 和高斯噪声 ε，均为 [B,H,A]；τ:[B]：

- xτ = τ ε + (1−τ)a。
- 监督速度 u = ε−a。
- 模型预测 vθ(xτ,τ,条件)，用 MSE 拟合 u。
- 推理从 noise 出发，Δτ=−1/steps，每次 x←x+Δτ·vθ，直到 0。

实现 flow_loss 时，若传 valid[B,H]，先对每个样本的有效步和动作维平均，再对 batch 平均；整条样本没有有效步时贡献 0。无 mask 时就是普通 MSE。

euler_sample 接收 velocity_fn，不改动输入 noise，不创建 autograd 图；steps≤0 抛 ValueError。第 k 次调用使用 τ=1−k/steps。

CFG 约定：v = v_uncond + β(v_positive−v_uncond)。β=0 为 uncond，β=1 为 positive。项目里的 uncond **只去掉 advantage 条件**，仍保留图像、任务和状态；两个分支必须使用相同的 xτ、τ。

## 分层提示

<details><summary>提示一：先检查端点</summary>

把 τ=0、τ=1 代入插值式，再对 τ 求导。这样可以自己推回目标速度和采样方向。

τ:[B] 应变成 [B,1,1] 才能和 [B,H,A] 相乘。
</details>

<details><summary>提示二：loss 和迭代</summary>

masked loss 的分母是每个样本“有效步数×A”，不能直接对补零后的整个 H 取 mean。

推理每一步都要调用 velocity_fn；测试使用已知常速度，最终应该精确走到目标动作。若越走越远，优先查符号。
</details>

## 口头追问与本地差异

1. 模型输出的是动作还是速度？action_out_proj 的形状说明了什么？
2. 训练一次前向为什么不需要展开全部 Euler 步？
3. advantage dropout 为什么支持 CFG？部署时怎样确保两个分支都实际构造出来？
4. β 大是否必然更好？你会比较哪些真机指标？

本题是基本 MSE。当前 JAX PiStar 的 compute_loss 还有随 τ 变化的样本权重，需单独读实际分支；不要说成用 advantage 数值乘 loss。PyTorch forward 与项目的 JAX 修改也不一定完全同步。
