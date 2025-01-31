import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def global_random_seed() -> int:
    return np.random.randint(0, 1000)


@pytest.fixture
def random_uniform() -> NDArray[np.float64]:
    return np.random.uniform(0, 100, size=10000)
