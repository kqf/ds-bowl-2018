import pytest

import torch

from model.model import UNet


@pytest.fixture(scope="module")
def batch(batch_size=128, n_channels=3, imsize=32):
    return torch.rand((batch_size, n_channels, imsize, imsize))


def test_model(batch):
    batch_size, n_channels, imsize, imsize = batch.shape
    model = UNet()
    assert model(batch).shape == (batch_size, 1, imsize, imsize)
