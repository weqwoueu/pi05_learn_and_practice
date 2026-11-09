import math
import pytest
import torch
from exercises.ex05_expert import time_embedding, prefix_suffix_mask, MiniActionExpert
from exercises.ex04_flow import make_flow_batch, flow_loss

def test_time_embedding_values_and_shapes():
    tau = torch.tensor([0., .5])
    actual = time_embedding(tau, 8)
    periods = torch.logspace(math.log10(.004), math.log10(4.), 4)
    angle = 2 * math.pi * tau[:, None] / periods
    expected = torch.cat([angle.sin(), angle.cos()], -1)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    assert time_embedding(torch.zeros(2, 5), 8).shape == (2, 5, 8)
    torch.testing.assert_close(actual[0], torch.tensor([0.] * 4 + [1.] * 4))
    with pytest.raises(ValueError):
        time_embedding(tau, 7)

def test_prefix_suffix_information_flow():
    valid = torch.tensor([[True, False, True]])
    mask = prefix_suffix_mask(valid, 2)
    expected = torch.tensor([[
        [True, False, True, False, False],
        [False, False, False, False, False],
        [True, False, True, False, False],
        [True, False, True, True, True],
        [True, False, True, True, True],
    ]])
    torch.testing.assert_close(mask, expected[:, None])
    assert mask[0, 0, 3, 4], "动作chunk内部双向，不能照搬GPT下三角"

def make_inputs():
    return torch.randn(2, 4, 16), torch.tensor([[True, True, False, False], [True, True, True, False]]), torch.randn(2, 5, 7)

def enable_time_condition(model):
    # 零初始化gate尚未更新时，初始输出不依赖tau；主动打开调制再测时间路径。
    with torch.no_grad():
        model.adaptive_norm.modulation.weight.normal_(std=.1)
        model.adaptive_norm.modulation.bias[-model.dim:].fill_(.5)

def test_expert_shape_and_complete_backward():
    model = MiniActionExpert(dim=16, heads=4, depth=2)
    enable_time_condition(model)
    prefix, valid, noisy = make_inputs()
    prefix.requires_grad_()
    noisy.requires_grad_()
    tau = torch.tensor([.2, .7], requires_grad=True)
    out = model(prefix, valid, noisy, tau)
    assert out.shape == noisy.shape
    out.square().mean().backward()
    for value in (prefix, noisy, tau):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
    assert model.action_in.weight.grad.abs().sum() > 0
    assert model.time_mlp[0].weight.grad.abs().sum() > 0

def test_prefix_cannot_read_actions_or_time():
    model = MiniActionExpert(dim=16, heads=4)
    enable_time_condition(model)
    prefix, valid, noisy = make_inputs()
    _, p1 = model(prefix, valid, noisy, torch.tensor([.2, .7]), return_prefix=True)
    _, p2 = model(prefix, valid, noisy + 9, torch.tensor([.8, .1]), return_prefix=True)
    torch.testing.assert_close(p1, p2)

def test_padding_cannot_affect_actions_but_valid_prefix_can():
    model = MiniActionExpert(dim=16, heads=4)
    prefix, valid, noisy = make_inputs()
    tau = torch.tensor([.2, .7])
    original = model(prefix, valid, noisy, tau)
    changed = prefix.clone()
    changed[~valid] += 999
    torch.testing.assert_close(model(changed, valid, noisy, tau), original)
    changed[valid] += torch.randn_like(changed[valid]) * 5
    assert not torch.allclose(model(changed, valid, noisy, tau), original)

def test_per_token_time_and_flow_training():
    model = MiniActionExpert(dim=16, heads=4)
    enable_time_condition(model)
    prefix, valid, noise = make_inputs()
    actions, tau = torch.randn_like(noise), torch.tensor([.3, .7])
    x, target = make_flow_batch(actions, noise, tau)
    times = tau[:, None].expand(2, 5)
    torch.testing.assert_close(model(prefix, valid, x, tau), model(prefix, valid, x, times))
    loss = flow_loss(model(prefix, valid, x, times), target)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.action_out.weight.grad.abs().sum() > 0
