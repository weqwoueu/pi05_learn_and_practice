import torch
import torch.nn.functional as F
from exercises.ex03_transformer import RMSNorm, GatedMLP, TransformerBlock, AdaptiveRMSNorm

def test_rms_does_not_center_and_uses_weight():
    layer = RMSNorm(3)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([1., 2., 3.]))
    x = torch.full((2, 4, 3), 2.)
    expected = x / torch.sqrt(torch.tensor(4. + layer.eps)) * layer.weight
    torch.testing.assert_close(layer(x), expected)
    assert (layer(x) > .99).all()

def test_rms_float32_moment_avoids_half_overflow():
    x = torch.full((1, 2, 4), 1000., dtype=torch.float16, requires_grad=True)
    out = RMSNorm(4)(x)
    assert out.dtype == x.dtype and torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.ones_like(out), atol=1e-3, rtol=1e-3)
    out.sum().backward()
    assert torch.isfinite(x.grad).all()

def test_gated_mlp_has_two_branches():
    model = GatedMLP(8, 12)
    x = torch.randn(2, 3, 8, requires_grad=True)
    expected = model.down_proj(F.gelu(model.gate_proj(x), approximate="tanh") * model.up_proj(x))
    torch.testing.assert_close(model(x), expected)
    model(x).square().mean().backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0

def test_block_prenorm_order():
    model = TransformerBlock(8, 2, 16)
    x = torch.randn(2, 3, 8)
    mask = torch.ones(2, 1, 3, 3, dtype=torch.bool).tril()
    first = x + model.attn(model.norm1(x), allowed=mask)
    expected = first + model.mlp(model.norm2(first))
    torch.testing.assert_close(model(x, mask), expected)

def test_zero_residual_branches_preserve_input():
    model = TransformerBlock(8, 2, 16)
    with torch.no_grad():
        model.attn.out_proj.weight.zero_()
        model.mlp.down_proj.weight.zero_()
    x = torch.randn(2, 3, 8, requires_grad=True)
    out = model(x)
    torch.testing.assert_close(out, x)
    out.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))

def test_adaptive_norm_zero_init_and_negative_gate():
    model = AdaptiveRMSNorm(4, 6)
    x, cond = torch.randn(2, 3, 4), torch.randn(2, 6)
    normalized, gate = model(x, cond)
    expected = x / torch.sqrt(x.square().mean(-1, keepdim=True) + model.eps)
    torch.testing.assert_close(normalized, expected)
    assert (gate == 0).all()
    with torch.no_grad():
        model.modulation.bias.copy_(torch.tensor([.5] * 4 + [2.] * 4 + [-2.] * 4))
    normalized, gate = model(x, cond)
    torch.testing.assert_close(normalized, expected * 1.5 + 2)
    assert (gate == -2).all(), "gate直接控制残差，不是sigmoid概率"

def test_adaptive_norm_per_token_condition():
    model = AdaptiveRMSNorm(4, 5)
    with torch.no_grad():
        model.modulation.weight.normal_(std=.2)
    x, cond = torch.randn(2, 3, 4), torch.randn(2, 3, 5)
    out, gate = model(x, cond)
    assert out.shape == gate.shape == x.shape
    changed = cond.clone()
    changed[:, 1] += 2
    out2, _ = model(x, changed)
    torch.testing.assert_close(out[:, 0], out2[:, 0])
    assert not torch.allclose(out[:, 1], out2[:, 1])
