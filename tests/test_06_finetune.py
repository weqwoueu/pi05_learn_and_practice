import copy
import torch
import torch.nn.functional as F
from exercises.ex06_finetune import AdapterModel, freeze_backbone, train_step, gradient_report

def test_freeze_clears_stale_grads_and_keeps_upstream_trainable():
    model = AdapterModel()
    model(torch.randn(3, 4)).sum().backward()
    freeze_backbone(model)
    for name, p in model.named_parameters():
        if name.startswith("backbone."):
            assert not p.requires_grad and p.grad is None
        else:
            assert p.requires_grad

def test_frozen_backbone_unchanged_adapter_still_updates():
    model = AdapterModel()
    freeze_backbone(model)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    loss = train_step(model, optimizer, torch.randn(5, 4), torch.randn(5, 2))
    assert isinstance(loss, float)
    report = gradient_report(model)
    for name, p in model.named_parameters():
        if name.startswith("backbone."):
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
            assert not report[name]
    assert report["adapter.weight"] and report["head.weight"]
    assert not torch.equal(model.adapter.weight, before["adapter.weight"])

def test_two_steps_match_reference_with_clipping_and_zeroing():
    model = AdapterModel()
    freeze_backbone(model)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.2)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.2)
    for _ in range(2):
        x, target = torch.randn(5, 4) * 10, torch.randn(5, 2) * 100
        model.eval()
        actual_loss = train_step(model, optimizer, x, target, max_grad_norm=.1)
        reference.train()
        ref_optimizer.zero_grad(set_to_none=True)
        expected_loss = F.mse_loss(reference(x), target)
        expected_loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), .1)
        ref_optimizer.step()
        assert model.training
        assert abs(actual_loss - expected_loss.item()) < 1e-4
        for p, ref in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, ref, atol=1e-6, rtol=1e-5)

def test_report_handles_none_zero_nan_and_nonzero():
    model = torch.nn.Linear(2, 1)
    assert gradient_report(model) == {"weight": False, "bias": False}
    model.weight.grad = torch.zeros_like(model.weight)
    model.bias.grad = torch.ones_like(model.bias)
    assert gradient_report(model) == {"weight": False, "bias": True}
    model.weight.grad[0, 0] = float("nan")
    assert not gradient_report(model)["weight"]
