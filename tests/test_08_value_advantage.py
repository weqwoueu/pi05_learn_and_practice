import math
import numpy as np
import pytest
import torch
from exercises.ex08_value_advantage import two_hot, value_cross_entropy, expected_value, n_step_advantage, label_advantages

def test_two_hot_edges_interpolation_and_mass():
    labels = torch.tensor([-2., -1., -.9975, -.5, -.0025, 0., 2.])
    out = two_hot(labels)
    assert out.shape == (7, 201)
    assert (out >= 0).all()
    torch.testing.assert_close(out.sum(-1), torch.ones(7))
    assert (out > 1e-5).sum(-1).max() <= 2
    torch.testing.assert_close(out[[0, 1], 0], torch.ones(2))
    torch.testing.assert_close(out[2, :2], torch.tensor([.5, .5]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[3, 100], torch.tensor(1.))
    torch.testing.assert_close(out[4, -2:], torch.tensor([.5, .5]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[[5, 6], -1], torch.ones(2))

def test_two_hot_custom_support():
    out = two_hot(torch.tensor([.25, .75]), atoms=3, vmin=0., vmax=1.)
    torch.testing.assert_close(out, torch.tensor([[.5, .5, 0.], [0., .5, .5]]))

def test_cross_entropy_stable_and_gradient():
    logits = torch.zeros(2, 201, requires_grad=True)
    loss = value_cross_entropy(logits, torch.tensor([-.9975, 0.]))
    torch.testing.assert_close(loss, torch.tensor(math.log(201.)))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] < 0
    assert logits.grad[1, -1] < 0
    extreme = torch.full((1, 201), -1000.)
    extreme[:, -1] = 1000
    loss = value_cross_entropy(extreme, torch.tensor([-1.]))
    torch.testing.assert_close(loss, torch.tensor(2000.))

def test_expectation_uses_probabilities():
    torch.testing.assert_close(expected_value(torch.zeros(2, 201)), torch.full((2,), -.5))
    logits = torch.full((2, 201), -100.)
    logits[0, 0], logits[1, -1] = 100., 100.
    torch.testing.assert_close(expected_value(logits), torch.tensor([-1., 0.]))

def test_nstep_terminal_and_bootstrap():
    values = np.array([-.8, -.6, -.3, -.1])
    rewards = np.array([-.25, -.25, -.25, 0.])
    np.testing.assert_allclose(n_step_advantage(values, rewards, 2), [0., .0, .05, .1], atol=1e-8)
    failed_rewards = np.array([-.25, -.25, -.25, -1.])
    np.testing.assert_allclose(n_step_advantage(values, failed_rewards, 2), [0., .0, -.95, -.9], atol=1e-8)
    np.testing.assert_allclose(n_step_advantage(values, rewards, 50), [.05, .1, .05, .1], atol=1e-8)

def test_nstep_empty_and_bad_inputs():
    assert n_step_advantage(np.array([]), np.array([]), 2).size == 0
    with pytest.raises(ValueError):
        n_step_advantage(np.ones(2), np.ones(3), 2)
    with pytest.raises(ValueError):
        n_step_advantage(np.ones(2), np.ones(2), 0)

def test_threshold_only_autonomous_and_human_always_positive():
    advantages = np.array([0., 1., 2., 3., 1000., -1000.])
    intervention = np.array([0, 0, 0, 0, 1, 1])
    positive, threshold = label_advantages(advantages, intervention)
    assert np.isclose(threshold, 2.1)
    np.testing.assert_array_equal(positive, [False, False, False, True, True, True])
    assert positive.dtype == np.bool_

def test_ties_and_all_human():
    positive, threshold = label_advantages(np.zeros(10), np.zeros(10, dtype=int))
    assert positive.all() and threshold == 0.
    positive, threshold = label_advantages(np.array([-9., 3.]), np.ones(2, dtype=int))
    assert positive.all() and threshold is None
