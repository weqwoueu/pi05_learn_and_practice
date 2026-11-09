import pytest
import torch
from exercises.ex04_flow import make_flow_batch, flow_loss, euler_sample, cfg_velocity

def test_flow_endpoints_and_velocity_sign():
    actions, noise = torch.randn(3, 4, 2), torch.randn(3, 4, 2)
    tau = torch.tensor([0., 1., .25])
    x, target = make_flow_batch(actions, noise, tau)
    torch.testing.assert_close(x[0], actions[0])
    torch.testing.assert_close(x[1], noise[1])
    torch.testing.assert_close(x[2], .25 * noise[2] + .75 * actions[2])
    torch.testing.assert_close(target, noise - actions)

def test_loss_per_sample_normalization_and_zero_mask():
    pred = torch.tensor([[[1., 1.], [100., 100.]],
                         [[3., 3.], [3., 3.]],
                         [[7., 7.], [7., 7.]]], requires_grad=True)
    valid = torch.tensor([[True, False], [True, True], [False, False]])
    loss = flow_loss(pred, torch.zeros_like(pred), valid)
    torch.testing.assert_close(loss, torch.tensor(10. / 3))
    loss.backward()
    assert (pred.grad[~valid] == 0).all()
    assert torch.isfinite(pred.grad).all()

def test_loss_without_mask():
    pred, target = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    torch.testing.assert_close(flow_loss(pred, target), (pred - target).square().mean())

@pytest.mark.parametrize("steps", [1, 5, 10])
def test_euler_oracle_reaches_action(steps):
    noise, actions = torch.randn(2, 4, 3, requires_grad=True), torch.randn(2, 4, 3)
    original = noise.detach().clone()
    calls = []
    def velocity(x, tau):
        calls.append(tau.clone())
        assert tau.shape == (2,) and not torch.is_grad_enabled()
        return original - actions
    out = euler_sample(velocity, noise, steps)
    torch.testing.assert_close(out, actions, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(noise, original)
    assert not out.requires_grad and len(calls) == steps
    torch.testing.assert_close(torch.stack(calls)[:, 0], 1 - torch.arange(steps) / steps)

def test_euler_invalid_steps():
    with pytest.raises(ValueError):
        euler_sample(lambda x, t: x, torch.zeros(1, 2, 3), 0)

def test_cfg_convention():
    positive, unconditional = torch.tensor([3., 5.]), torch.tensor([1., 2.])
    torch.testing.assert_close(cfg_velocity(positive, unconditional, 0), unconditional)
    torch.testing.assert_close(cfg_velocity(positive, unconditional, 1), positive)
    torch.testing.assert_close(cfg_velocity(positive, unconditional, 2), torch.tensor([5., 8.]))
