import time
import click
import torchvision
from contextlib import contextmanager
from pathlib import Path
from model.data import PatchedDataset
from model.data import CellsDataset
from model.model import build_model
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

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.RandomRotation(5),
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.RandomCrop(224, padding=10),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dirs = [p for p in Path(path).iterdir() if p.is_dir()]
    dataset = CellsDataset(dirs[:5], transforms=train_transform)
    plot_cells(*zip(*dataset))

    patched = PatchedDataset(dataset)
    model = build_model(max_epochs=2)
    with timer("Train the model"):
        model.fit(patched)

    with timer("Predict the labels"):
        preds = model.predict(patched)

    imgs, masks = zip(*patched)
    plot_cells(imgs, masks, preds)


if __name__ == '__main__':
    main()
