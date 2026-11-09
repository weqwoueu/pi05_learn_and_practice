import random
import numpy as np
import pytest
import torch

torch.set_num_threads(1)

@pytest.fixture(autouse=True)
def reproducible():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
