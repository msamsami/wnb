import numpy as np
import pytest


@pytest.fixture
def global_random_seed():
    return np.random.randint(0, 1000)
