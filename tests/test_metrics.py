import pytest
import numpy as np

from model.metrics import iou, iou_approx


@pytest.mark.parametrize("true_labels, logits", [
    (np.random.randn(2, 25, 25) > 0.5, np.random.randn(2, 25, 25)),
    (np.random.randn(3, 25, 25) > 0.5, np.random.randn(2, 25, 25)),
    (np.random.randn(2, 25, 25) > 0.5, np.random.randn(3, 25, 25)),
    (np.zeros((2, 25, 25)), np.random.randn(3, 25, 25)),
])
def test_calculates_metrics(true_labels, logits):
    y_true = true_labels.astype(int)
    assert iou(y_true, logits) == 0.0


@pytest.mark.parametrize("true_labels, logits", [
    (np.random.randn(3, 25, 25) > 0.5, np.random.randn(3, 25, 25)),
    (np.random.randn(3, 25, 25) > 0.5, np.random.randn(3, 25, 25)),
    (np.random.randn(3, 25, 25) > 0.5, np.random.randn(3, 25, 25)),
])
def test_calculates_approx_metrics(true_labels, logits):
    y_true = true_labels.astype(int)
    assert iou_approx(y_true, logits, padding=1) > 0.0


@pytest.mark.parametrize("true_labels, logits", [
    (np.zeros((1, 25, 25)), np.ones((1, 25, 25))),
    (np.zeros((3, 25, 25)), np.random.randn(3, 25, 25)),
])
def test_calculates_minimum_approx_metrics(true_labels, logits):
    y_true = true_labels.astype(int)
    assert iou_approx(y_true, logits) == 0.


def test_calculates_maximum_approx_metrics():
    y_true = np.eye(25, 25)[None, :, :]
    # Convert [0, 1] -> [-999, 999]
    logits = (y_true - 0.5) * 999 * 2
    assert iou_approx(y_true, logits) == 1.
