"""03 Transformer block与时间调制。参数构造已提供，重点实现forward。"""
import torch
from torch import nn
from exercises.ex02_attention import MultiHeadAttention

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        """最后维RMS归一化，不减均值；平方均值用float32，输出回到x.dtype。"""
        x_dtype = x.dtype #先记录x原来的数据类型
        x = x.float() #转类型
        x_norm =  (x / (((x*x).mean(dim=-1,keepdim = True) + self.eps).sqrt())) #RMS归一化公式
        x_weigth = x_norm * self.weight #自动广播相乘
        return x_weigth.to(x_dtype) #转回原来的x.dtype后返回
        

class GatedMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        """down_proj(gelu(gate_proj(x), approximate='tanh') * up_proj(x))。
        | 名称 | 含义 |
        |---|---|
        | `dim` | 输入、输出的特征数量，也就是 `D` |
        | `hidden` | 中间层的特征数量，通常比 `D` 大 |
        | `gate_proj` | 生成门控分支的特征 |
        | `up_proj` | 生成另一条分支的特征 |
        | `down_proj` | 把中间特征转换回 `D` 维 |                  
                   x：[B,N,8]
                  /        \
           gate_proj       up_proj
               ↓              ↓
           [B,N,12]        [B,N,12]
               ↓              │
             GELU             │
               └─── ×(乘法) ──┘
                       ↓
                   [B,N,12]
                       ↓
                   down_proj
                       ↓
                   [B,N,8]"""
        #用一下开头公式即可
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate='tanh')*self.up_proj(x))

        


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = GatedMLP(dim, hidden)

    def forward(self, x, allowed=None):
        """PreNorm：先Attention残差，再MLP残差；维度不变。
        实际上就是作两次残差链接
        x ───────────────────┐
        │                    │
        ↓                    │
        norm1 → Attention    │
             ↓               │
             └──── 相加 ←────┘
                    ↓
                    h
                    
        h ───────────────────┐
        │                    │
        ↓                    │
        norm2 → GatedMLP     │
             ↓               │
             └──── 相加 ←────┘
                    ↓
                   输出"""
        h = x + self.attn(self.norm1(x), allowed=allowed)
        return h + self.mlp(self.norm2(h))
        

class AdaptiveRMSNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.modulation = nn.Linear(cond_dim, 3 * dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, cond):
        """x:[B,N,D]；cond:[B,C]或[B,N,C]。
        基础RMS无独立可训练weight。输出(normalized*(1+scale)+shift, gate)。
        三分量顺序scale/shift/gate。gate不做sigmoid；本函数不加残差。
        就是在原来RMSNorm的基础上不要用那个全是1的weights，而是用(normalized*(1+scale)+shift, gate)来做输出
        cond
          ↓ modulation
        [ scale 的 D 个数 | shift 的 D 个数 | gate 的 D 个数 ]
        然后用.chunk分三个就行了"""
        x_dtype = x.dtype #这题不转回去也行
        x = x.float()
        normalized = (x / (((x*x).mean(dim=-1,keepdim = True) + self.eps).sqrt())) #RMSNorm公式
        cond_lin = self.modulation(cond)
        if cond.shape == (cond.shape[0],cond.shape[1]):
            cond_lin = cond_lin.unsqueeze(1)
        scale , shift ,gate = cond_lin.chunk(3 ,dim = -1) #x.chunk(3,dim = -1)表示最后一维分三个，因为self.modulation = nn.Linear(cond_dim, 3 * dim)
        return normalized*(1+scale)+shift , gate
        
