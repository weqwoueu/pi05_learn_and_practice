"""02 Attention。mask统一True=允许关注；禁止nn.MultiheadAttention/SDPA代写核心。"""
import torch
from torch import nn
from exercises.ex00_tensors import split_heads, merge_heads

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor,
                                 v: torch.Tensor,
                                 allowed: torch.Tensor | None = None):
    """q:[B,H,Nq,d], k:[B,H,Nk,d], v:[B,H,Nk,dv]。
    allowed可广播至[B,H,Nq,Nk]。返回(out, weights)。
    全mask行的weights/out必须为0且梯度有限；不使用dropout。
    | 符号 | 含义 | 举例 |
    |---|---|---|
    | `B` | batch size，一批有几条样本 | 2 条样本 |
    | `h` | 注意力头数 | 4 个头 |
    | `Nq` | query 的数量 | 3 个 query |
    | `Nk` | key 的数量，也是 value 的数量 | 5 对 key/value |
    | `d` | 每个头中，单个 query/key 向量的长度 | 8 个数字 |
    | `dv` | 每个头中，单个 value 向量的长度 | 6 个数字 |
    """
    scores = (q@k.transpose(-2,-1))/(q.shape[-1]**0.5) #x.sqrt()是x为张量的函数不能用
    if allowed is None:
        weights = scores.softmax(dim=-1)
    else:
        masked_scores = scores.masked_fill(~allowed, float("-inf"))
        if_vaild = allowed.any(dim=-1,keepdim=True) #判断每一行是否至少有一个True，有就当前行返回True，没有就False，保留维度方便广播
        safe_scores = masked_scores.masked_fill(~if_vaild, 0.0) #如果有全-inf行也就是全遮挡行，-inf全部变0.0，否则softmax有除0错误
        weights = safe_scores.softmax(dim=-1)
        #此时得到全遮挡行为全为1/Nk(和为1)用原来的allowed遮挡即可得到全0
        weights = weights.masked_fill(~allowed, 0.0) #其他mask的已经为0.0，再mask一次掉全False情况
    out = weights@v
    return out,weights


def make_attention_mask(query_valid: torch.Tensor, key_valid: torch.Tensor,
                        causal: bool = False) -> torch.Tensor:
    """valid:[B,Nq]/[B,Nk] -> bool [B,1,Nq,Nk]。
    causal=True仅支持Nq==Nk，否则抛ValueError；只允许key索引<=query。
    用&（且）也就是都True才True
    """
    mask = query_valid.unsqueeze(-1)&key_valid.unsqueeze(-2)
    if causal is True:
        if mask.shape[-1] == mask.shape[-2]:
            mask = mask.tril()  #tril()是指左下True保留包括对角线，其他True不保留，也就是只允许key索引<=query
        else:
            raise ValueError
    mask = mask.unsqueeze(-3)
    return mask
    



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, context=None, allowed=None):
        """query:[B,Nq,D]，context:[B,Nk,D]；None表示self attention。
        返回[B,Nq,D]；复用本题函数，可复用ex00的拆头合头。
        """
        if context is None:
            context = query
        Q = self.q_proj(query)
        K = self.k_proj(context)
        V = self.v_proj(context)
        #拆头
        Q_s_h= split_heads(Q,self.num_heads)
        K_s_h= split_heads(K,self.num_heads)
        V_s_h= split_heads(V,self.num_heads)
        #过注意力
        out,weights = scaled_dot_product_attention(Q_s_h,K_s_h,V_s_h,allowed)
        #合头
        return self.out_proj(merge_heads(out)) #合头后过的投影层可以收集各头信息


        
