"""08 Value/Advantage，默认201个支撑点[-1,0]。"""
import numpy as np
import torch

def two_hot(values: torch.Tensor, atoms=201, vmin=-1.0, vmax=0.0):
    """values:[B]，超范围clip；输出[B,atoms]，按相邻支撑点插值。"""
    if atoms < 2 or vmax <= vmin:
        raise ValueError("至少需要两个支撑点，且 vmax 必须大于 vmin")

    # 1. 把范围外的标签截到端点；使用浮点数计算插值权重。
    # 半精度输入先升为 float32，float64 输入则保留精度。
    dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    clipped = values.to(dtype=dtype).clamp(vmin, vmax)

    # 2. 把“价值数值”映射成“连续下标”，范围为 [0, atoms-1]。
    # 默认间距为 1/200=0.005：-1 对应下标 0，0 对应下标 200。
    # 例如 -0.9975 对应约 0.5，表示落在下标 0 和 1 的中间。
    position = (clipped - vmin) / (vmax - vmin) * (atoms - 1)
    position = position.clamp(0, atoms - 1)  # 防止浮点舍入导致下标越界
    left = position.floor().long()  # 下标需要整数类型
    right = (left + 1).clamp(max=atoms - 1)

    # 3. 小数部分是右侧权重，剩余部分是左侧权重，两者之和为 1。
    # position=3.2 时：第 3 个下标放 0.8，第 4 个下标放 0.2。
    right_weight = position - left
    left_weight = 1 - right_weight

    # 4. 为每条样本建立一行分布，将权重加到各自的左右下标。
    # scatter_add_ 沿 dim=1（支撑点维）累加；下标和权重都是 [B,1]。
    labels = torch.zeros(
        (values.shape[0], atoms), device=values.device, dtype=dtype,
    )
    labels.scatter_add_(1, left[:, None], left_weight[:, None])
    labels.scatter_add_(1, right[:, None], right_weight[:, None])

    # 恰好位于支撑点时，小数部分为 0，退化成 one-hot。
    # 最右端点的 left/right 都是 atoms-1，累加 1 和 0，仍保留总权重 1。
    return labels

def value_cross_entropy(logits, labels):
    """logits:[B,201], labels:[B]；稳定log_softmax，与two_hot求交叉熵。
    | 输入 | 形状 | 含义 |
    |---|---|---|
    | `logits` | `[B,201]` | 模型为 201 个价值支撑点输出的分数，还不是概率 |
    | `labels` | `[B]` | 每条样本的真实价值标签，例如 `-0.9975` |"""
    target = two_hot(labels) #把真实标签做成two-hot 分布)比如-0.9975对应target = [0.5, 0.5, 0, ..., 0]，也就是正确答案所在地
    log_probe = logits.log_softmax(dim = -1) # 它在数学上相当于先 softmax，再取自然对数也就是log(p₀)，log(p₁)pi是模型预测的201个位置的第i个位置概率
    loss = - (target * log_probe) # 例如loss = −0.5 × log(p₀) − 0.5 × log(p₁)，答案所在地方的预测概率越接近1，loss越小
    loss_sum = loss.sum(-1) #得到每一条样本的loss
    loss = loss_sum.mean()
    return loss



def expected_value(logits):
    """logits:[B,201] -> [B]，softmax概率乘支撑点求和。
    计算价值"""
    probe = logits.softmax(dim=-1) # 模型预测分数softmax得到模型预测概率
    point_201 = torch.linspace(-1 , 0 , 201 , dtype = logits.dtype , device = logits.device) # 201 个支撑点均匀分布在 [-1,0]
    v = probe * point_201
    return v.sum(dim= -1)



def n_step_advantage(values: np.ndarray, rewards: np.ndarray,
                     lookahead=50) -> np.ndarray:
    """输入单个完整episode的[T]数组，gamma=1。
    A_t=sum rewards[t:t+N]+V[t+N]-V[t]；跨终点时bootstrap=0。
    空轨迹返回空数组；N<=0或长度不一致抛ValueError；不跨episode。
    """
    if lookahead <= 0:
        raise ValueError("lookahead 必须大于 0")
    if len(values) != len(rewards):
        raise ValueError("values 和 rewards 的长度必须一致")

    length = len(values)
    # 每帧输出一个优势值；用浮点类型，避免整数输入导致小数被截断。
    # 空轨迹时数组为空，下面的循环不会执行，直接返回空数组。
    advantages = np.zeros(length, dtype=np.float64)
    for t in range(length):
        future = t + lookahead
        end = min(future, length)
        # 累计从当前帧开始的最多 N 步奖励，包含窗口内的终点奖励。
        reward_sum = rewards[t:end].sum(dtype=np.float64)
        # 未来位置仍在本 episode 内才读取 V；到达或越过终点时取零。
        bootstrap = values[future] if future < length else 0.0
        advantages[t] = reward_sum + bootstrap - values[t]

    return advantages

def label_advantages(advantages: np.ndarray, intervention: np.ndarray,
                     top_percent=30.0):
    """输入所有待标注rollout帧，不含跳过value的全程人工demo。
    仅用intervention==0的帧计算np.percentile(...,100-top_percent)。
    人工帧直接True，自主帧>=阈值为True；返回(bool mask, threshold)。
    全人工时threshold=None，mask全True；ties不强行凑30%。
    """
    # 布尔索引区分自主帧与人工帧；人工帧直接 positive。
    autonomous = intervention == 0
    positive = (intervention == 1).copy()

    # 全人工时没有自主 adv 可计算分位数，返回全 True 和 None。
    # 空输入也会返回空的 bool 数组和 None。
    if not autonomous.any():
        return positive, None

    # 只用全部自主帧计算一个阈值，人工帧的 adv 不影响阈值。
    # 默认取第 70 百分位：数值较大的自主帧被标为 positive。
    threshold = np.percentile(advantages[autonomous], 100 - top_percent)

    # 将自主帧的判断结果填回原位置，人工帧的 True 保持不变。
    # 用 >= 保留所有与阈值同分的帧，不强行凑出精确的 30%。
    positive[autonomous] = advantages[autonomous] >= threshold
    return positive, threshold
