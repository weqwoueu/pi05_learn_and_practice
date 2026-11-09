"""04 Flow Matching与CFG。tau=1为噪声，tau=0为动作。"""
from collections.abc import Callable
import torch

def make_flow_batch(actions: torch.Tensor, noise: torch.Tensor,
                    tau: torch.Tensor):
    """actions/noise:[B,H,A]，tau:[B] -> (x_tau, target_velocity)。
    公式xτ = τ ε + (1−τ)a。noise为ε"""
    x_tua = tau[:,None,None]*noise + (1-tau[:,None,None])*actions #记得广播就行，维度不同actions/noise:[B,H,A]，tau:[B]
    # 其实就是两点线性插值，tua就是x在点1到0上，y0就是noise，y1就是actions，y(x)就是x_tua，不过方向反过来
    # target_velocity就是 actions - noise
    target_velocity = noise - actions  #注意tua是反的(1—->0)，所以目标速度也是反的
    return x_tua , target_velocity



def flow_loss(pred: torch.Tensor, target: torch.Tensor,
              valid: torch.Tensor | None = None):
    """[B,H,A]的MSE；valid:[B,H]时仅按有效位置与动作维平均。
    每个样本先独立平均，再在batch平均；没有有效步的样本贡献0。
    逐元素平方误差 [B,H,A]
    ↓ 忽略无效步，沿 H、A 求和
    每条样本的误差总和 [B]
    ↓ 除以各自的有效元素数量，处理分母为零
    每条样本的平均误差 [B]
    ↓ 对 batch 求平均
    最终 loss：一个标量 
    """
    loss_raw = (pred - target)*(pred - target)
    if valid is None:
        loss_mask_sum = loss_raw.sum(dim=-1).sum(dim = -1)
        loss_mask_sum_mean = loss_mask_sum / (pred.shape[1]*pred.shape[2]) #注意是全部元素数量不是一个pred.shape[1]
    else:
        loss_mask_sum = loss_raw.masked_fill(~valid[:,:,None] , 0.0 ).sum(dim=-1).sum(dim = -1)
        valid_sum = valid.sum(dim=-1)*pred.shape[2]
        loss_mask_sum_mean = loss_mask_sum / valid_sum.clamp_min(1) #此处保证最小为1，不除0，
    loss = loss_mask_sum_mean.mean()
    return loss


    

def euler_sample(velocity_fn: Callable, noise: torch.Tensor,
                 steps: int = 10) -> torch.Tensor:
    """velocity_fn(x:[B,H,A], tau:[B])->同形状；从tau=1到0。
    不改动输入noise，steps<=0抛ValueError，推理禁止构建autograd图。
    新位置 = 当前位置 + 时间步长 × 当前速度
    x = x + Δtau × velocity_fn(x, tau)
    tau = 1时
    """
    if steps <= 0:
        raise ValueError
    with torch.no_grad(): #推理禁止构建autograd图
        tau = torch.ones(noise.shape[0]) #创建全一的形状和noise第一维一样的tau
        dt = -1/steps
        x = noise
        for i in range(steps): #循环要确保同步更新tau作为传入值才能计算velocity进行去噪，这就是tau的意义
                x = x + dt*velocity_fn(x,tau )
                tau  = tau + dt
        return x



    
    

def cfg_velocity(v_positive: torch.Tensor, v_unconditional: torch.Tensor,
                 beta: float) -> torch.Tensor:
    """这里unconditional仅指去掉advantage，图像/语言任务/状态保留。
    v_positive	带 Advantage: positive 条件时预测的速度
    v_unconditional	去掉 advantage 条件时预测的速度
    beta	引导强度，一个数
    公式 v = v_unconditional + beta × (v_positive − v_unconditional)"""
    v = v_unconditional + beta * (v_positive - v_unconditional) # 实际上就是加上v_positive相对于v_unconditional的变化乘一个调整参数
    return v 

    #                  同一个 x_t、time
    #                    /          \
    #           带 positive 条件    无 advantage 条件
    #                  ↓                  ↓
    #              v_positive          v_uncond
    #                    \              /
    #                     CFG 组合速度
    #                           ↓
    #                     更新同一份 x_t
    
# 不是两条分支各自生成完整动作，再把最终动作混合。 它在每个去噪步骤中组合速度，然后更新一条共同的动作轨迹。
