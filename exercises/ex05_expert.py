"""05 综合题：共享宽度的教学动作专家，不是完整PaliGemma双专家复刻。
图像/文本编码器已由prefix embedding替代；不下载大模型权重。
"""
import math
import torch
import numpy as np
from torch import nn
from exercises.ex03_transformer import AdaptiveRMSNorm, RMSNorm, TransformerBlock

def time_embedding(tau: torch.Tensor, dim: int,
                   min_period: float = 4e-3, max_period: float = 4.0):
    """tau:[B]或[B,H]；dim必须为偶数且>=4。
    period=min_period*(max_period/min_period)^linspace(0,1,dim/2)
    拼接sin(2*pi*tau/period), cos(...)；输出先sin后cos[...,dim]。
    检查 dim，不符合要求时抛 ValueError。
    生成 dim // 2 个周期，使用浮点数，并放在 tau.device 上。
    给 tau 增加最后一个维度。
    计算角度，再分别取 sin、cos。
    沿最后一维拼接并返回。"""
    if dim< 4 or dim%2 != 0:
        raise ValueError
    pi = math.pi
    period = min_period*(max_period/min_period)**torch.linspace(0,1,dim//2,device=tau.device,dtype=torch.float32,) #此处传入整数不能dim/2
    angle = 2*pi*tau.unsqueeze(-1)/period
    sin = angle.sin()
    cos = angle.cos()
    return torch.cat([sin,cos] , dim = -1)

    


def prefix_suffix_mask(prefix_valid: torch.Tensor, horizon: int):
    """prefix_valid:[B,P] -> allowed:[B,1,P+H,P+H]。
    有效prefix之间互看，prefix不可看动作，动作可看有效prefix和所有动作。

    . prefix 和 suffix 是什么？
    [图像、任务、状态的特征 | 带噪动作的特征]
    └──── prefix ────┘   └── suffix ──┘
    - prefix：已经编码好的观测和任务信息，共 P 个 token。
    - suffix：动作 token，共 H 个，一个对应一步动作
    参数	            含义
    prefix_valid	    [B,P] 的布尔张量，表示哪些 prefix token 有效
    horizon	            动作步数，也就是 H  

    给 prefix_valid 后面拼上 H 个 True，得到全部 token 的有效性 [B,P+H]。
    用 query 有效性与 key 有效性做 &，生成 [B,P+H,P+H]。
    把“prefix 行、动作列”这一块设为 False。
    增加 head 维度，返回 [B,1,P+H,P+H]。
    """
    p = prefix_valid.shape[1]
    prefix_valid = torch.cat([prefix_valid,torch.ones(prefix_valid.shape[0],horizon,dtype=torch.bool)],dim=1)
    mask_raw = prefix_valid.unsqueeze(-1) & prefix_valid.unsqueeze(-2) 
    mask_raw[:,:p,p:] = False #让p行前(不包括p)p列后（包括p）为False
    """   p0 p1 │ a0 a1 a2
       p0  1  0 │  0  0  0 这里
       p1  0  0 │  0  0  0 这里
          ──────┼──────────
       a0  1  0 │  1  1  1
       a1  1  0 │  1  1  1
       a2  1  0 │  1  1  1
    """
    return mask_raw.unsqueeze(1)


    

class MiniActionExpert(nn.Module):
    def __init__(self, action_dim=7, dim=32, heads=4, depth=2):
        super().__init__()
        self.dim = dim
        self.action_in = nn.Linear(action_dim, dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                      nn.Linear(dim, dim), nn.SiLU())
        self.adaptive_norm = AdaptiveRMSNorm(dim, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, 2 * dim) for _ in range(depth)
        ])
        self.output_norm = RMSNorm(dim)
        self.action_out = nn.Linear(dim, action_dim)

    def forward(self, prefix, prefix_valid, noisy_actions, tau,
                return_prefix=False):
        """prefix:[B,P,D], prefix_valid:[B,P], noisy_actions:[B,H,A]。
        tau:[B]或[B,H]（为RTC扩展保留）。输出[B,H,A]。
        步骤见题面；本题只在入口做一次时间调制，完整pi05在各层调制。
        return_prefix=True时额外返回更新后的prefix，供检查信息泄漏。
        [观测和任务的特征 ||带噪动作的特征]
        └ prefix(前缀) ──┘└─suffix(后缀)┘
        | 参数 | 形状 | 含义 |
        |---|---|---|
        | `prefix` | `[B,P,D]` | 已编码好的图像、任务、状态特征 |
        | `prefix_valid` | `[B,P]` | 哪些 prefix token 有效 |
        | `noisy_actions` | `[B,H,A]` | 当前带噪动作 |
        | `tau` | `[B]` 或 `[B,H]` | 去噪时间 |
        | `return_prefix` | 布尔值 | 是否额外返回更新后的 prefix |
        带噪动作 → action_in ───────────────┐
                                         ↓
        时间 → time_embedding → time_mlp → 条件调制与残差
                                     ↓
                         与 prefix 拼接
                                  ↓
                      带 mask 的 Transformer
                                  ↓
                           取动作 token
                                  ↓
                      output_norm → action_out
                                  ↓
                             预测速度"""
        action_tokens = self.action_in(noisy_actions) #把带噪动作转换成模型内部的特征
        time_tokens = time_embedding(tau , self.dim) #正余弦编码先把时间数字变成向量
        cond = self.time_mlp(time_tokens) #MLP 再学习如何利用这些时间特征
        modulated , gate = self.adaptive_norm(action_tokens , cond) #让时间条件调整动作特征
        action_tokens = action_tokens + gate*modulated #残差链接，这样动作 token 就带上了“当前处于哪个去噪阶段”的信息
        tokens = torch.cat([prefix,action_tokens],dim=1) #cat([],...)记得要[]代表合成一个列表
        # prefix：       [B,P,D]
        # action_tokens：[B,H,D]
                    # ↓ 沿 token 维拼接
        # tokens：       [B,P+H,D]
        allowed = prefix_suffix_mask(prefix_valid,noisy_actions.shape[1])
        tokens_trans = tokens
        for blocks in self.blocks: #带着 mask 通过 Transformer，一个动作 token 可以从图像、任务和状态中收集信息，也能参考其他动作位置的特征，帮助整段动作协调
            tokens_trans = blocks(tokens_trans , allowed = allowed)
        # 注意这里的 self.blocks是一个列表，要循环遍历列表
        # Transformer 输出仍是 [B,P+H,D]。前 P 个是 prefix，后 H 个是动作：
        tokens = tokens_trans[:,tokens_trans.shape[1] - noisy_actions.shape[1]:,:]
        # 后 H 个 token：[B,H,D]
        #   ↓ self.output_norm
        #   [B,H,D]
        vol = self.action_out(self.output_norm(tokens))
        #   ↓ self.action_out.output_norm
        # 预测速度：   [B,H,A]
        if return_prefix is False:
            return vol
        else:
            return vol , tokens_trans[:,  :tokens_trans.shape[1] - noisy_actions.shape[1],:]
    """补充
    参数来源

    相机图像
    ↓ 图像预处理、视觉编码器、投影
    图像特征 ──────────────────────┐
                                │
    任务文字 + 归一化并分桶后的状态   │
    ↓ 构造 prompt                │
    ↓ tokenizer → token IDs      │
    ↓ Embedding                  │
    文本特征 ──────────────────────┤
                     ↓
            沿 token 维拼接
                     ↓
            prefix：[B,P,D]
    
            然后我的动作专家接受prefix, prefix_valid, noisy_actions, tau,

            根据当前观测、任务和去噪进度，这段带噪动作应该怎样变化？
            得到他的预测速度
            其实就是euler_sample里面的速度函数
            然后进行欧拉迭代
            如果要做cfg的话就先把两个速度做cfg_velocity
            然后再给euler_sample做迭代
    在 π0.5 先把 τ 做多频率正余弦编码，再经过两层带 SiLU 的 MLP，得到与动作专家宽度一致的条件向量。
    （这之前和前面的一样，区别后面是区别）该条件传入动作专家各层的 adaRMS（前面的代码只有一个adaRMS，π0.5各层都有另一个），
    此外cond不和action_tokens一起进block各层的adaRMS，比如attention一个adaRMS，mlp层一个adaRMS
    adaRMS做的就是先把x正常rmsnorm，然后让编码好的时间条件过mlp投影成动作专家宽度一致的条件向量，然后三倍投影成scale、shift、gate，然后对rms后的x调制，公式见ex03
    在 Attention 和 MLP 前分别生成 scale、shift、gate：scale 和 shift 调制 RMS 归一化后的特征，gate 控制相应分支输出的残差贡献。
    各层共用同一时间条件，但调制层参数各自独立；它不通过新增时间 token 来注入

    """
·
















        
        
