"""07 Training-time RTC：0..6控制步，与去噪时间tau分开。"""
from collections.abc import Callable
import torch

def sample_delays(batch_size: int, max_delay=6, generator=None):
    """CPU整数Tensor[B]，均匀采样0..max_delay，包含上限。"""
    delays = torch.randint(0,max_delay+1,(batch_size,) , generator=generator)
    return delays

def make_rtc_batch(actions, noise, tau, delays):
    """actions/noise:[B,H,A], tau:[B], delays:[B], 0<=delay<H。
    返回(x_tau, target, token_time[B,H], prefix_mask[B,H])。
    前缀干净且时间0，后缀按tau加噪，target仍为noise-actions。
    actions、noise：[B,H,A]
    tau：           [B]
    delays：        [B]
    """
    for d in delays:
        if not 0<= d < actions.shape[1]:
            raise ValueError
    position = torch.arange(actions.shape[1])
    prefix_mask = position.unsqueeze(0) < delays.unsqueeze(1)
    # 位置 [1,H] < delays [B,1] → prefix_mask [B,H]
    zero = torch.zeros(tau.shape[0])
    token_time = torch.where(prefix_mask , zero.unsqueeze(-1) , tau.unsqueeze(-1))
    #构造 token_time：前缀位置取 0，其他位置取该样本的 tau。可以考虑用 torch.where
    t = token_time.unsqueeze(-1)
    x_tau = t * noise + (1-t) * actions #老样子做x_tau
    target = noise - actions
    return x_tau , target , token_time , prefix_mask


def rtc_loss(pred, target, prefix_mask):
    """仅监督后缀，前缀是上一步的动作，先每样本按后缀步数与动作维平均，再平均batch。
    任一样本全部为前缀时抛ValueError（训练时delay<H）。
    | 输入 | 形状 | 含义 |
    |---|---|---|
    | `pred` | `[B,H,A]` | 模型预测的速度 |
    | `target` | `[B,H,A]` | 第二题返回的 `noise - actions` |
    | `prefix_mask` | `[B,H]` | `True` 是已知前缀，`False` 是需要预测的后缀 |

    计算逐元素平方误差 (pred - target)²，形状 [B,H,A]。
    根据 prefix_mask 把前缀误差置零。这里 True 就是要屏蔽的位置。
    沿 H、A 求和，得到每条样本的误差总和 [B]。
    每条样本除以自己的 后缀步数 × A。
    对 batch 求平均，返回一个标量。
    """
    if prefix_mask.all(dim=-1).any():
        raise ValueError
    loss_raw = (target - pred)*(target - pred)
    loss_masked = loss_raw.masked_fill(prefix_mask.unsqueeze(-1) , 0.0)
    loss_sum = loss_masked.sum(-1).sum(-1)
    loss_mean = loss_sum / ((~prefix_mask).sum(-1) * pred.shape[-1])
    loss_mean = loss_mean.mean()
    return loss_mean


    

def rtc_sample(velocity_fn: Callable, noise, known_actions, delays, steps=10):
    """velocity_fn(x, token_time[B,H])->[B,H,A]。
    每步固定前缀并使前缀更新量为0，后缀Euler更新；不修改输入。
    delays可为0..H（全前缀允许用于推理测试），无autograd图。

    ① 将 x 的前缀覆盖成 known_actions
    ② 构造 token_time：前缀为 0，后缀为当前 tau
    ③ 调用模型，得到速度 v
    ④ 将 v 的前缀部分清零
    ⑤ 执行 x = x + dt × v
    ⑥ 更新时间
    """
    if steps <= 0:
        raise ValueError("steps 必须大于 0")
    horizon = noise.shape[1]
    if ((delays < 0) | (delays > horizon)).any():
        raise ValueError("推理延迟必须满足 0 <= delay <= H")

    with torch.no_grad():
        # 每条样本前 delay 步为已知动作；mask：[B,H]。
        position = torch.arange(horizon, device=noise.device)
        prefix_mask = position[None, :] < delays.to(noise.device)[:, None]
        action_mask = prefix_mask.unsqueeze(-1)  # [B,H,1]，广播到动作分量
        x = noise.clone()
        dt = -1.0 / steps

        for i in range(steps):
            # 每次调用模型前恢复已知前缀，不修改传入的两个动作张量。
            x = torch.where(action_mask, known_actions, x)
            tau = torch.full(
                noise.shape[:2], 1 - i / steps,
                device=noise.device, dtype=noise.dtype,
            )
            token_time = tau.masked_fill(prefix_mask, 0.0)

            velocity = velocity_fn(x, token_time)
            # 时间为零不保证模型预测速度为零，所以显式禁止前缀更新。
            velocity = velocity.masked_fill(action_mask, 0.0)
            x = x + dt * velocity

        # 确保最终返回的前缀与已知动作一致，包括 delay=H 的情况。
        return torch.where(action_mask, known_actions, x)

def next_chunk_offset(observation_step: int, switch_step: int,
                      new_horizon: int) -> int:
    """按绝对控制步编号对齐。返回应跳过的新chunk动作数。
    switch<observation，或offset>=new_horizon（结果已耗尽）时抛ValueError。
    | 参数 | 含义 |
    |---|---|
    | `observation_step` | 获取观测、启动新动作推理时的绝对控制步编号 |
    | `switch_step` | 实际开始执行新动作序列时的绝对控制步编号 |
    | `new_horizon` | 新动作序列的总长度 |"""
    if  switch_step < observation_step :
        raise ValueError("切换时间早于观测时间")
    d_step = switch_step - observation_step #延迟长度,也就是应跳过的步数
    if d_step >= new_horizon:
        raise ValueError("新序列全给你跳了，那延迟很高了")
    return d_step




    
