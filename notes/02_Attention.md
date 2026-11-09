# 02 手写 Multi-Head Attention

需要查函数或类时，打开 [API 速查](Torch与Python_API速查.md)，按顶部题号表阅读。可以看着 API 写第一遍。

实现：[ex02_attention.py](../exercises/ex02_attention.py)。检查：`uv run python run.py 02`。建议 60—90 分钟。先完成 00。

## 要完成什么

A. Scaled Dot-Product Attention：

- Q:[B,h,Nq,d]，K:[B,h,Nk,d]，V:[B,h,Nk,dv]。
- scores=QKᵀ/√d，沿最后一维 softmax；输出 weights:[B,h,Nq,Nk] 和 out:[B,h,Nq,dv]。
- allowed 是可广播的 bool mask，**True 表示允许关注**。
- 被遮挡位置权重为 0。整行都被遮挡时，权重和输出均为 0，反向梯度也必须有限。
- 不使用 dropout。

B. 构造 mask：query_valid[B,Nq] 与 key_valid[B,Nk] → [B,1,Nq,Nk]。可选 causal 仅支持 Nq=Nk，允许 key 位置≤query 位置。

C. MultiHeadAttention：Q 来自 query，K/V 来自 context；context=None 时为 self attention。完成投影、拆头、注意力、合头、输出投影。构造函数已给出。

限制：核心实现不能直接调用 nn.MultiheadAttention 或 F.scaled_dot_product_attention；测试会用它们作参照。可以复用 00。

## 分层提示

<details><summary>提示一：先算一个 head</summary>

矩阵乘法最后两维负责 token 间的点积。K 需要交换最后两维；缩放因子来自每个 head 的 d，不是整个 d_model。

mask 的行是 query，列是 key；只屏蔽 key padding 不能自动保证 padding query 的输出为零。
</details>

<details><summary>提示二：全遮挡行的 NaN</summary>

全部置为 -inf 后直接 softmax，会得到 NaN。先找出整行都不允许的位置，在 softmax 前给这些行一个有限占位值；softmax 后再将不允许的位置清零。

仅在最后 nan_to_num 可能掩盖前向问题而留下反向 NaN。测试会检查梯度。
</details>

## 口头追问

1. Q、K、V 的作用和形状是什么？cross attention 的 Nq/Nk 能否不同？
2. 为什么除以 √d？softmax 应沿哪个维度？
3. 因果 mask 和 padding mask 各解决什么问题？
4. 动作 chunk 内必须因果吗？到第 05 题验证。
5. 当前教学实现是 MHA；真实 Gemma 的 GQA 中 Q 头数与 KV 头数有什么区别？先能指出区别，不要求本题实现 GQA。
