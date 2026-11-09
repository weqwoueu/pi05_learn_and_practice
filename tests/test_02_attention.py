import math
import pytest
import torch
import torch.nn.functional as F
from exercises.ex02_attention import scaled_dot_product_attention, make_attention_mask, MultiHeadAttention

def test_attention_matches_pytorch_oracle():
    q, k, v = torch.randn(2, 3, 4, 5), torch.randn(2, 3, 6, 5), torch.randn(2, 3, 6, 7)
    mask = torch.rand(2, 1, 4, 6) > .3
    mask[..., 0] = True
    out, weights = scaled_dot_product_attention(q, k, v, mask)
    reference = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(out, reference)
    assert weights.shape == (2, 3, 4, 6)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3, 4))
    assert (weights.masked_select(~mask.expand_as(weights)) == 0).all()

def test_scale_is_head_dimension():
    q = torch.tensor([[[[2., 0., 0., 0.]]]])
    k = torch.tensor([[[[2., 0., 0., 0.], [0., 0., 0., 0.]]]])
    v = torch.tensor([[[[3.], [8.]]]])
    out, weights = scaled_dot_product_attention(q, k, v)
    expected = torch.softmax(torch.tensor([2., 0.]), dim=-1)
    torch.testing.assert_close(weights.flatten(), expected)
    torch.testing.assert_close(out.flatten(), (expected * torch.tensor([3., 8.])).sum().reshape(1))

def test_all_masked_rows_are_zero_and_backward_finite():
    q = torch.randn(1, 2, 3, 4, requires_grad=True)
    k = torch.randn(1, 2, 4, 4, requires_grad=True)
    v = torch.randn(1, 2, 4, 6, requires_grad=True)
    allowed = torch.ones(1, 1, 3, 4, dtype=torch.bool)
    allowed[:, :, 1] = False
    out, weights = scaled_dot_product_attention(q, k, v, allowed)
    assert torch.isfinite(out).all() and torch.isfinite(weights).all()
    assert (out[:, :, 1] == 0).all() and (weights[:, :, 1] == 0).all()
    out.square().sum().backward()
    for tensor in (q, k, v):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    assert (q.grad[:, :, 1] == 0).all()

def test_padding_and_causal_mask():
    query = torch.tensor([[True, True, False]])
    key = torch.tensor([[True, False, True]])
    actual = make_attention_mask(query, key, causal=True)
    expected = torch.tensor([[[[True, False, False], [True, False, False], [False, False, False]]]])
    torch.testing.assert_close(actual, expected)
    cross = make_attention_mask(torch.ones(2, 3, dtype=torch.bool), torch.ones(2, 5, dtype=torch.bool))
    assert cross.shape == (2, 1, 3, 5) and cross.all()
    with pytest.raises(ValueError):
        make_attention_mask(torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool), causal=True)

def test_multihead_cross_attention_matches_torch():
    model = MultiHeadAttention(12, 3)
    reference = torch.nn.MultiheadAttention(12, 3, batch_first=True, bias=False)
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat([model.q_proj.weight, model.k_proj.weight, model.v_proj.weight]))
        reference.out_proj.weight.copy_(model.out_proj.weight)
    query, context = torch.randn(2, 4, 12), torch.randn(2, 6, 12)
    expected, _ = reference(query, context, context, need_weights=False)
    torch.testing.assert_close(model(query, context), expected, atol=1e-6, rtol=1e-5)
    expected, _ = reference(query, query, query, need_weights=False)
    torch.testing.assert_close(model(query), expected, atol=1e-6, rtol=1e-5)

def test_masked_context_cannot_change_output():
    model = MultiHeadAttention(8, 2)
    query, context = torch.randn(1, 3, 8), torch.randn(1, 4, 8)
    allowed = torch.tensor([[[[True, True, False, False]]]]).expand(1, 1, 3, 4)
    changed = context.clone()
    changed[:, 2:] += 100
    torch.testing.assert_close(model(query, context, allowed), model(query, changed, allowed))

def test_bad_heads_rejected():
    with pytest.raises(ValueError):
        MultiHeadAttention(7, 2)
