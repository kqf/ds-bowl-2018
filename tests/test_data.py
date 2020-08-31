import random
from model.data import CellsDataset
from model.model import train_transform

from pathlib import Path


def test_dataset(path="data/cells/"):
    folders = random.choices(list(Path(path).iterdir()), k=2)
    data = CellsDataset(folders, transform=train_transform())
    X, y = next(iter((data)))

    x_channels, *x_shape = X.shape
    y_shape = y.shape

    assert tuple(x_shape) == tuple(y_shape)
    assert x_channels == 3
