import pytest
import random
from model.data import PatchedDataset, CellsDataset, GenericDataset
from model.model import train_transform

from pathlib import Path


@pytest.fixture
def data(path="data/cells/"):
    folders = random.choices(list(Path(path).iterdir()), k=2)
    cells = CellsDataset(folders)
    return PatchedDataset(cells)


def test_data(data):
    X, y = next(iter((data)))

    x_channels, *x_shape = X.shape
    y_shape = y.shape

    assert tuple(x_shape) == tuple(y_shape)
    assert x_channels == 3


def test_generic(path="data/cells/"):
    folders = random.choices(list(Path(path).iterdir()), k=2)
    data = GenericDataset(folders, transform=train_transform())
    X, y = next(iter((data)))

    x_channels, *x_shape = X.shape
    y_shape = y.shape

    assert tuple(x_shape) == tuple(y_shape)
    assert x_channels == 3
