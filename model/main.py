import click
from pathlib import Path
from model.data import PatchedDataset
from model.data import CellsDataset
from model.model import build_model


@click.group()
def main():
    pass


@main.command()
@click.option("--path", type=click.Path(exists=True), default="data/cells")
def train(path):
    dirs = [p for p in Path(path).iterdir() if p.is_dir()]
    dataset = PatchedDataset(CellsDataset(dirs))

    model = build_model()
    model.fit(dataset)


if __name__ == '__main__':
    main()
