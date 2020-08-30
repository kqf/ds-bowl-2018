import click
import cv2

from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm
from PIL import Image

from itertools import zip_longest, product, chain
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import (pad, to_tensor, normalize,
                                               hflip, vflip, crop)


def calcuate_bboxes(im_shape, patch_size):
    """Calculate bound boxes based on image shape and size of the bounding box
    given by `patch_size`"""
    h, w = im_shape
    ph, pw = patch_size

    steps_h = chain(range(0, h - ph, ph), [h - ph])
    steps_w = chain(range(0, w - pw, pw), [w - pw])

    return product(steps_h, steps_w)


class PatchedDataset(Dataset):
    """Creates patches of cells.
    Parameters
    ----------
    base_dataset: CellsDataset
        Dataset of cells
    patch_size: tuple of ints (default=(256, 256))
        The size of each patch
    random_flips: bool (default=False)
        If true, patches and masks will be randomly flipped horizontally and
        vertically.
    padding: int (default=16)
        Amount of paddding around each image and mask
    """

    def __init__(self,
                 base_dataset,
                 patch_size=(256, 256),
                 random_flips=False,
                 padding=16):
        super().__init__()
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.patch_size_expanded = (patch_size[0] + 2 * padding,
                                    patch_size[1] + 2 * padding)
        self.padding = padding
        self.random_flips = random_flips

        coords = []
        for idx, (_, mask) in enumerate(self.base_dataset):
            w, h = mask.size
            bboxes = calcuate_bboxes((h, w), self.patch_size)
            idx_bboxes = list(zip_longest([], bboxes, fillvalue=idx))
            coords.extend(idx_bboxes)
        self.coords = coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        img_idx, (i, j) = self.coords[idx]
        cell, mask = self.base_dataset[img_idx]
        h, w = self.patch_size_expanded

        cell = pad(cell, self.padding, padding_mode='reflect')
        mask = pad(mask, self.padding, padding_mode='reflect')

        cell = crop(cell, i, j, h, w)
        mask = crop(mask, i, j, h, w)

        if self.random_flips:
            if random.random() < 0.5:
                cell = hflip(cell)
                mask = hflip(mask)

            if random.random() < 0.5:
                cell = vflip(cell)
                mask = vflip(mask)

        cell = to_tensor(cell)
        mask = torch.as_tensor((np.array(mask) == 255).astype('float32'))

        # mean and std of imagenet
        cell = normalize(cell, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return cell, mask


class CellsDataset(Dataset):
    """Constructs cell dataset"""

    def __init__(self, sample_dirs):
        super().__init__()
        self.sample_dirs = sample_dirs

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        cell_fn = (sample_dir / 'images' / sample_dir.name).with_suffix('.png')
        mask_fn = sample_dir / 'mask.png'

        cell, mask = Image.open(cell_fn).convert('RGB'), Image.open(mask_fn)
        assert cell.size == mask.size
        return cell, mask


class GenericDataset(Dataset):
    def __init__(self, sample_dirs, transform=None):
        super().__init__()
        self.sample_dirs = sample_dirs
        self.transform = transform

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        cell_fn = (sample_dir / 'images' / sample_dir.name).with_suffix('.png')
        mask_fn = sample_dir / 'mask.png'

        # Read an image with OpenCV
        image, mask = cv2.imread(cell_fn), cv2.imread(mask_fn)

        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, mask


def combine_masks(mask_root_dir):
    mask_output = mask_root_dir / 'mask.png'
    if mask_output.exists():
        return

    mask_fn_iter = mask_root_dir.glob('masks/*.png')
    img = Image.open(next(mask_fn_iter))
    for fn in mask_fn_iter:
        mask = Image.open(fn)
        img.paste(mask, (0, 0), mask)

    img.save(mask_output)


@click.command()
@click.option("--path", type=click.Path(exists=True), default="data/cells")
def main(path):
    # Combine masks into one
    samples_dirs = list(d for d in Path(path).iterdir() if d.is_dir())
    with Pool() as pool, tqdm(total=len(samples_dirs)) as pbar:
        for _ in tqdm(pool.imap_unordered(combine_masks, samples_dirs)):
            pbar.update()


if __name__ == '__main__':
    main()
