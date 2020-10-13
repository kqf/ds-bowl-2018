import pytest
import numpy as np

from model.metrics import iou, iou_approx


@pytest.mark.parametrize("true_labels, predicted", [
    (np.random.randn(2, 25, 25) > 0.5, np.random.randn(2, 25, 25)),
    (np.random.randn(3, 25, 25) > 0.5, np.random.randn(2, 25, 25)),
    (np.random.randn(2, 25, 25) > 0.5, np.random.randn(3, 25, 25)),
    (np.zeros((2, 25, 25)), np.random.randn(3, 25, 25)),
])
def test_calculates_metrics(true_labels, predicted):
    y_true = true_labels.astype(int)
    assert iou(y_true, predicted) == 0.0


@pytest.mark.parametrize("true_labels, predicted", [
    (np.random.randn(3, 25, 25) > 0.5, np.random.randn(3, 25, 25)),
    (np.random.randn(3, 25, 25) > 0.5, np.random.randn(3, 25, 25)),
    (np.random.randn(3, 25, 25) > 0.5, np.random.randn(3, 25, 25)),
    (np.zeros((3, 25, 25)), np.random.randn(3, 25, 25)),
])
def test_calculates_approx_metrics(true_labels, predicted):
    y_true = true_labels.astype(int)
    assert np.isnan(iou_approx(y_true, predicted))


@pytest.mark.skip("TODO: fix me")
@pytest.mark.parametrize("true_labels, predicted", [
    (np.ones((3, 25, 25)), np.ones((3, 25, 25))),
])
def test_calculates_maximum_approx_metrics(true_labels, predicted):
    y_true = true_labels.astype(int)
    assert iou_approx(y_true, predicted) == 25
