"""01 状态输入处理。精确语义、边界和tokenizer说明见题面。"""
from collections.abc import Callable
import numpy as np

def normalize_quantile(x, q01, q99, eps=1e-6):
    """最后一维对应状态维。与仓库一致：加eps，不自动clip。"""
    if q99-q01 <= 1e-2 :
        raise ValueError("q99与q01过于接近")
    y = 2*(x-q01)/(q99-q01+eps)-1
    return y

def unnormalize_quantile(y, q01, q99, eps=1e-6):
    """本题只处理与统计量同维的数据，返回原单位。"""
    x = (y+1)*(q99-q01+eps)/2+q01
    return x

def discretize_state(state: np.ndarray, bins: int = 256) -> np.ndarray:
    """对齐 np.digitize(state, linspace(-1,1,bins+1)[:-1])-1。
    特别注意：当前仓库低于-1会得到-1，并未显式clip。
    要求默认 256 桶。−1→0，0→128，1→255；低于−1→−1，高于1→255
    """
    state_256 = np.digitize(state,np.linspace(-1,1,bins+1)[:-1])-1
    #linspace是吧-1到1分成bins+1个数，也就是bins个区间，此时为[0,1,2...256]
    #[:-1]代表取到当前列表最后一个元素前的列表（因为达到或超过末边界返回 len(bins)）
    #np.digitze分state到[0,1,2...256]这个列表的区间里面，state是值在-1到1的长度随机的列表
    #生成长度为state长度的每一个state的值在256bin里面的区间序号
    return state_256

    

def build_prompt(task: str, normalized_state: np.ndarray,
                 advantage: str | None = None) -> str:
    """Task: ..., State: <空格分隔的整数>[, Advantage: ...];\nAction: """
    clean_task = task.strip().replace("_"," ").replace("\n" , " ")
    clean_state =" ".join(map(str,discretize_state(normalized_state)))
    if advantage == None:
        return f"Task: {clean_task}, State: {clean_state};\nAction: "
    else:
        return f"Task: {clean_task}, State: {clean_state}, Advantage: {advantage};\nAction: "



def encode_prompt(prompt: str, encode: Callable[[str], list[int]],
                  max_length: int) -> tuple[np.ndarray, np.ndarray]:
    """encode是注入的分词回调（含BOS等由回调负责），不是数字直接当token。
    返回 int64 IDs [L] 和 bool mask [L]。右侧补0，超长截断。
    mask按实际长度生成，不能靠 token!=0；长度<=0抛ValueError。
    """
    if max_length<=0:
        raise ValueError
    raw_id = encode(prompt)
    mask = [True]*max_length #生成max_length长度的全ture列表
    mask_len = [True]*len(raw_id) 
    mask_False = [False]*max_length
    if raw_id == []:
        return np.array(raw_id + [0]*max_length,dtype =np.int64), np.array(mask_False,dtype=np.bool_)
    else:
        if len(raw_id) >= max_length:
            return np.array(raw_id[:max_length],dtype =np.int64),np.array(mask,dtype=np.bool_)
        elif len(raw_id) < max_length:
            return np.array(raw_id + [0]*(max_length-len(raw_id)),dtype =np.int64),np.array(mask_len+[False]*(max_length-len(raw_id)),dtype=np.bool_)
       
#优化写法多用[:n]别老是加加加

def encode_prompt(
    prompt: str,
    encode: Callable[[str], list[int]],
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_length <= 0:
        raise ValueError("max_length 必须大于 0")

    raw_ids = encode(prompt)
    n = min(len(raw_ids), max_length)

    ids = np.zeros(max_length, dtype=np.int64)
    mask = np.zeros(max_length, dtype=np.bool_)

    ids[:n] = raw_ids[:n]  #这个好用
    mask[:n] = True

    return ids, mask
