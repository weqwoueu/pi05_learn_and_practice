"""00 热身。题面见 notes/00_热身.md。"""
import torch

def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """[B,N,D] -> [B,H,N,D/H]；D不能整除H时抛ValueError。"""
    if x.shape[2] % num_heads != 0:
        raise ValueError("D要被H整除")
    x =  x.reshape(x.shape[0],x.shape[1],num_heads,x.shape[2]//num_heads).transpose(1,2) #注意此处是//因为这个代表dim，维度是int不是浮点
    if x.is_contiguous():
        return x
    else:
        return x.contiguous().view(x.shape[0],x.shape[1],x.shape[2],x.shape[3])
    

def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """[B,H,N,d] -> [B,N,H*d]，支持非连续张量。"""
    #reshape只能合或分相邻维度
    #错误示范
    #return x.reshape(x.shape[0],x.shape[1]*x.shape[3],x.shape[2]).transpose(1,2)
    #return x.transpose(1,2).reshape(x.shape[0],x.shape[1],x.shape[2]*x.shape[3])  #reshape只能合或分相邻维度（还是错）
    #错在x.shape[0],x.shape[1]还有后面的序号还是原来的x的顺序，x.transpose(1,2)只是新生成一个x，不改变原来的x
    return x.transpose(1,2).reshape(x.shape[0],x.shape[2],x.shape[1]*x.shape[3]) #正确的要按照原来的x序号而不是x.transpose(1,2)的序号

def masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """x:[B,N,D], valid:[B,N] bool -> [B,D]；全无效样本返回零。"""
    x_maksed = x*valid.unsqueeze(-1)
    #return x_maksed.sum(dim=1)/valid.sum(dim = 1 ,keepdim=True)
    #此处可能有0/0的情况，题目要去此时输出0，因此解决办法不是if0/0，而是用clamp_min去限制分母最小为1
    return x_maksed.sum(dim=1)/valid.sum(dim = 1 ,keepdim=True).clamp_min(1)
