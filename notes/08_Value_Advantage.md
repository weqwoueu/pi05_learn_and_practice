# 08 Value 分布与 Advantage 标注

需要查函数或类时，打开 [API 速查](Torch与Python_API速查.md)，按顶部题号表阅读。可以看着 API 写第一遍。

实现：[ex08_value_advantage.py](../exercises/ex08_value_advantage.py)。检查：`uv run python run.py 08`。建议 45—60 分钟。

## A. 201 维价值头

支撑点 z_i 均匀覆盖 [-1,0]，共 201 个，间隔 0.005。201 维输出是 logits，不是 201 个动作。

- two_hot：标量标签分配给相邻两个支撑点，权重由距离决定，总和为 1；恰好在支撑点上就是 one-hot；超范围 clip。支持自定义 atoms/vmin/vmax。
- value_cross_entropy：two-hot 目标乘 log_softmax(logits) 后负求和，再 batch 平均；不要先 softmax 再 log，极端 logits 会数值不稳定。
- expected_value：概率 softmax(logits) 与支撑点加权求和，返回 [B]。推理时不是直接对 logits 求平均。

例子：−0.9975 位于 −1 和 −0.995 正中间，两者各 0.5。均匀概率的期望是 −0.5。

## B. N 步 Advantage

输入是**单个 episode**的预测 value 与奖励数组，默认 N=50、γ=1：

A_t = Σ rewards[t:t+N] + V[t+N] − V[t]。

若 t+N 超出 episode 范围，bootstrap 取 0，但范围内的终点奖励仍要累计。不要跨 episode 取未来值；也不要把未来 value 每一帧全加进去。空轨迹返回空数组，N≤0 或两数组长度不同报错。

这是一步一帧输出 A_t，只是每帧使用向前 N 步的窗口；不是每 50 帧只输出一个标签。

## C. positive 标注

label_advantages 输入全部待标 rollout 帧：

- intervention==1 的人工帧直接 positive。
- 只用 intervention==0 的自主帧，算全局第 70 百分位阈值；adv≥阈值为 positive。
- 不按 episode 各选 30%；也不把人工帧纳入这个阈值。
- 同分时可能超过 30%，不强行精确截取数量。
- 全是人工帧时全部 positive，threshold=None。
- 全程人工成功 demo 若本来跳过 Value 推理，应在外部直接保留 positive，不必塞进这个函数。

对你的第一轮混合数据，判断依据是**帧级 intervention**，不是只看整条轨迹叫“失败”还是“接管成功”。接管成功轨迹若含 intervention=0 的自主前缀，该前缀也参与分位数标注；全人工的 50 条 SFT demo 则直接 positive。

## 分层提示

<details><summary>提示一：先把标签换成连续下标</summary>

将标签从 [vmin,vmax] 映射至 [0,atoms−1]。取左右下标，按小数部分分配权重。端点尤其要查，避免右端点全部权重丢失。

Advantage 可以先写直观循环，四天冲刺不必优先追求向量化。
</details>

<details><summary>提示二：别把终点奖励和 bootstrap 混为一谈</summary>

reward[t:min(t+N,T)] 仍包含窗口内终点；只有 V[t+N] 越界时归零。

先取自主帧子集求 np.percentile，再将结果映射回完整数组。人工极大或极小的 advantage 都不应改变自主阈值。
</details>

## 口头追问

- 分布价值为什么最终仍能得到一个标量？two-hot 相比硬分桶保留了什么？
- Value 误差如何影响 advantage 与 positive 标签？
- 为什么成功/失败是轨迹结局，而 advantage 可以给失败轨迹里的部分动作较高评价？
- 你的价值训练标签、reward_label、预测 V 分别是什么？不要当成同一数组。
- 第一轮没有自主成功轨迹，会限制什么？哪些结论需要额外对照实验支持？
