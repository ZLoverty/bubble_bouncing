from bounce import Bubble
import pytest
import numpy as np

@pytest.fixture
def bubble():
    return Bubble(1e-3, U=0.1)

def test_Oseen_wake(bubble):
    assert bubble.Oseen_wake(np.random.rand(10, 3)).shape[1] == 3

