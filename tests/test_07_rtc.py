import pytest
import torch
from exercises.ex07_rtc import sample_delays, make_rtc_batch, rtc_loss, rtc_sample, next_chunk_offset

def test_delay_range_includes_zero_and_six():
    a = sample_delays(2048, 6, torch.Generator().manual_seed(3))
    b = sample_delays(2048, 6, torch.Generator().manual_seed(3))
    assert a.shape == (2048,) and a.dtype == torch.long
    assert set(a.tolist()) == set(range(7))
    torch.testing.assert_close(a, b)
    assert (sample_delays(8, 0) == 0).all()

def test_rtc_clean_prefix_and_per_token_time():
    actions, noise = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    tau, delays = torch.tensor([.25, .75]), torch.tensor([0, 2])
    x, target, times, prefix = make_rtc_batch(actions, noise, tau, delays)
    expected_prefix = torch.tensor([[False, False, False, False], [True, True, False, False]])
    torch.testing.assert_close(prefix, expected_prefix)
    torch.testing.assert_close(times, torch.tensor([[.25] * 4, [0., 0., .75, .75]]))
    torch.testing.assert_close(x[prefix], actions[prefix])
    mixed = tau[:, None, None] * noise + (1 - tau[:, None, None]) * actions
    torch.testing.assert_close(x[~prefix], mixed[~prefix])
    torch.testing.assert_close(target, noise - actions)

@pytest.mark.parametrize("delay", [-1, 4])
def test_training_delay_must_leave_suffix(delay):
    with pytest.raises(ValueError):
        make_rtc_batch(torch.zeros(1, 4, 2), torch.ones(1, 4, 2), torch.tensor([.5]), torch.tensor([delay]))

def test_suffix_loss_normalized_per_sample():
    pred = torch.tensor([[[100.], [100.], [100.], [1.]], [[3.], [3.], [3.], [3.]]], requires_grad=True)
    prefix = torch.tensor([[True, True, True, False], [False] * 4])
    loss = rtc_loss(pred, torch.zeros_like(pred), prefix)
    torch.testing.assert_close(loss, torch.tensor(5.))
    loss.backward()
    assert (pred.grad[prefix] == 0).all()
    with pytest.raises(ValueError):
        rtc_loss(pred, torch.zeros_like(pred), torch.ones_like(prefix))

def test_sampling_prefix_is_fixed_on_every_model_call():
    noise = torch.randn(2, 4, 3, requires_grad=True)
    original = noise.detach().clone()
    known, goal = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    known_before = known.clone()
    delays = torch.tensor([0, 2])
    prefix = torch.arange(4)[None, :] < delays[:, None]
    calls = []
    def velocity(x, times):
        assert not torch.is_grad_enabled()
        torch.testing.assert_close(x[prefix], known[prefix])
        assert (times[prefix] == 0).all()
        calls.append(times.clone())
        v = original - goal
        v[prefix] = 999
        return v
    out = rtc_sample(velocity, noise, known, delays, steps=5)
    torch.testing.assert_close(out[prefix], known[prefix])
    torch.testing.assert_close(out[~prefix], goal[~prefix], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(noise, original)
    torch.testing.assert_close(known, known_before)
    assert not out.requires_grad and len(calls) == 5

def test_sampling_all_prefix_and_invalid_steps():
    noise, known = torch.randn(1, 4, 2), torch.randn(1, 4, 2)
    out = rtc_sample(lambda x, t: torch.ones_like(x), noise, known, torch.tensor([4]), 3)
    torch.testing.assert_close(out, known)
    with pytest.raises(ValueError):
        rtc_sample(lambda x, t: x, noise, known, torch.tensor([0]), 0)

def test_chunk_alignment_uses_absolute_steps():
    assert next_chunk_offset(20, 50, 50) == 30
    assert next_chunk_offset(20, 24, 50) == 4
    assert next_chunk_offset(20, 20, 50) == 0
    with pytest.raises(ValueError):
        next_chunk_offset(20, 19, 50)
    with pytest.raises(ValueError):
        next_chunk_offset(20, 70, 50)
