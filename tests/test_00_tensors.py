import pytest
import torch
from exercises.ex00_tensors import split_heads, merge_heads, masked_mean

def test_head_order_and_roundtrip():
    x = torch.arange(2 * 3 * 8).reshape(2, 3, 8)
    y = split_heads(x, 2)
    assert y.shape == (2, 2, 3, 4)
    torch.testing.assert_close(y[1, 1, 2], x[1, 2, 4:])
    torch.testing.assert_close(merge_heads(y), x)

def test_split_rejects_indivisible():
    with pytest.raises(ValueError):
        split_heads(torch.zeros(2, 3, 7), 2)

def test_noncontiguous_merge_and_gradient():
    x = torch.randn(2, 5, 3, 4, requires_grad=True)
    heads = x.transpose(1, 2)
    assert not heads.is_contiguous()
    out = merge_heads(heads)
    torch.testing.assert_close(out, x.reshape(2, 5, 12))
    out.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))

def test_masked_mean_all_invalid_and_gradient():
    x = torch.tensor([[[1., 2.], [999., 999.], [5., 6.]],
                      [[3., 4.], [5., 6.], [7., 8.]]], requires_grad=True)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    out = masked_mean(x, mask)
    torch.testing.assert_close(out, torch.tensor([[3., 4.], [0., 0.]]))
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert (x.grad[~mask] == 0).all()
    torch.testing.assert_close(x.grad[0, 0], torch.full((2,), .5))
