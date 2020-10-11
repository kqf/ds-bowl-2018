import numpy as np
import pytest
from model.metrics import iou


@pytest.fixture
def true_labels(n=20):
    return (np.random.randn(n, 256, 256) > 0.5).astype(int)


@pytest.fixture
def predicted(n=10):
    return (np.random.randn(n, 256, 256) > 0.5).astype(int)


def test_calculates_metrics(true_labels, predicted):
    assert iou(true_labels, predicted) == 0.0
