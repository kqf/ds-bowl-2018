import pytest
import numpy as np

from model.metrics import iou


@pytest.mark.parametrize("true_labels, predicted", [
    (np.random.randn(2, 256, 256) > 0.5, np.random.randn(2, 256, 256) > 0.5),
    (np.random.randn(3, 256, 256) > 0.5, np.random.randn(2, 256, 256) > 0.5),
    (np.random.randn(2, 256, 256) > 0.5, np.random.randn(3, 256, 256) > 0.5),
    (np.zeros((2, 256, 256)), np.random.randn(3, 256, 256) > 0.5),
])
def test_calculates_metrics(true_labels, predicted):
    assert iou(true_labels.astype(int), predicted.astype(int)) == 0.0
