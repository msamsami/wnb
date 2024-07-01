import numpy as np
import pytest


@pytest.fixture
def global_random_seed():
    return np.random.randint(0, 1000)


@pytest.fixture
def random_uniform():
    return np.random.uniform(0, 100, size=10000)
