import click

from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm
from PIL import Image


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
