import time
import click
from contextlib import contextmanager
from pathlib import Path
from model.data import PatchedDataset
from model.data import CellsDataset
from model.model import build_model


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
    dataset = PatchedDataset(CellsDataset(dirs))

    model = build_model()
    with timer("Train the model"):
        model.fit(dataset)


if __name__ == '__main__':
    main()
