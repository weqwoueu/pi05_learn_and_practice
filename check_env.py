"""Verify the interview environment; this file contains no exercise answers."""
import io
import json
import platform

import einops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch


def main():
    torch.manual_seed(0)
    inputs = torch.from_numpy(np.ones((2, 4), dtype=np.float32))
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model.weight.detach().clone()
    loss = model(inputs).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad).all()
    optimizer.step()
    assert not torch.equal(before, model.weight.detach())
    assert einops.rearrange(inputs, "b d -> d b").shape == (4, 2)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    assert buffer.tell() > 100
    print(json.dumps({
        "status": "PASS",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pytest": pytest.__version__,
        "matplotlib": matplotlib.__version__,
        "einops": einops.__version__,
        "device": "cpu",
        "verified": ["numpy_tensor_bridge", "forward_backward", "optimizer_update", "einops", "plot_render"],
    }, indent=2))


if __name__ == "__main__":
    main()