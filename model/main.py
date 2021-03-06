import time
import click
from contextlib import contextmanager
from pathlib import Path
from model.data import CellsDataset
from model.model import build_model, train_transform, test_transform
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


def infer(model, dataset, title):
    print(f"Infering for {title} set")
    plot_cells(*zip(*dataset))
    with timer("Predict the labels"):
        preds = model.predict(dataset)

    imgs, masks = zip(*dataset)
    plot_cells(imgs, masks, preds)


@main.command()
@click.option("--path", type=click.Path(exists=True), default="data/cells")
def train(path):
    dirs = [p for p in Path(path).iterdir() if p.is_dir()]
    dataset = CellsDataset(dirs[:5], transform=train_transform())
    plot_cells(*zip(*dataset))

    model = build_model(max_epochs=2)
    with timer("Train the model"):
        model.fit(dataset)

    infer(model, dataset, "train")

    # Infer for all types of images
    model.set_params(batch_size=1)
    test = CellsDataset(dirs[:2], transform=test_transform())
    infer(model, test, "test")


if __name__ == '__main__':
    main()
