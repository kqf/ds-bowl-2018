import pytest
import numpy as np

from model.metrics import iou


@pytest.mark.parametrize("true_labels, predicted", [
    (np.random.randn(2, 25, 25) > 0.5, np.random.randn(2, 25, 25)),
    (np.random.randn(3, 25, 25) > 0.5, np.random.randn(2, 25, 25)),
    (np.random.randn(2, 25, 25) > 0.5, np.random.randn(3, 25, 25)),
    (np.zeros((2, 25, 25)), np.random.randn(3, 25, 25)),
])
def test_calculates_metrics(true_labels, predicted):
    y_true = true_labels.astype(int)
    assert iou(y_true, predicted) == 0.0
