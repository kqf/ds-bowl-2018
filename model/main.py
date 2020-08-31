import time
import click
import torchvision
from contextlib import contextmanager
from pathlib import Path
from model.data import PatchedDataset
from model.data import CellsDataset
from model.data import GenericDataset
from model.model import build_model, train_transform
from model.vis import plot_cells


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print("{color}[{name}] done in {et:.0f} s{nocolor}".format(
        name=name, et=time.time() - t0,
        color='\033[1;33m', nocolor='\033[0m'))


@click.group()
def main():
    pass


@main.command()
@click.option("--path", type=click.Path(exists=True), default="data/cells")
def train(path):
    dirs = [p for p in Path(path).iterdir() if p.is_dir()]
    dataset = GenericDataset(dirs[:5], transform=train_transform())
    plot_cells(*zip(*dataset))

    model = build_model(max_epochs=2)
    with timer("Train the model"):
        model.fit(dataset)

    with timer("Predict the labels"):
        preds = model.predict(dataset)

    imgs, masks = zip(*dataset)
    plot_cells(imgs, masks, preds)


if __name__ == '__main__':
    main()
