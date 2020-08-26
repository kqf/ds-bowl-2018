import torch
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt


def tensor2img(t, padding=16):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    mu = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    img = to_pil_image(t.mul(std).add_(mu))
    w, h = img.size
    return img.crop((padding, padding, w - padding, h - padding))


def plot_mask_cell(true_mask,
                   predicted_mask,
                   cell,
                   suffix,
                   ax1,
                   ax2,
                   ax3,
                   padding=16):
    """Plots a single cell with a its true mask and predicuted mask"""
    for ax in [ax1, ax2, ax3]:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    ax1.imshow(true_mask[padding:-padding, padding:-padding], cmap='viridis')
    ax1.set_title('True Mask - {}'.format(suffix))
    ax2.imshow(
        predicted_mask[padding:-padding, padding:-padding], cmap='viridis')
    ax2.set_title('Predicted Mask - {}'.format(suffix))
    ax3.imshow(tensor2img(cell, padding=padding))
    ax3.set_title('Image - {}'.format(suffix))
    return ax1, ax2, ax3


def plot_mask_cells(mask_cells, padding=16):
    fig, axes = plt.subplots(len(mask_cells), 3, figsize=(12, 10))
    for idx, (axes, mask_cell) in enumerate(zip(axes, mask_cells), 1):
        ax1, ax2, ax3 = axes
        true_mask, predicted_mask, cell = mask_cell
        plot_mask_cell(
            true_mask, predicted_mask, cell,
            'Type {}'.format(idx),
            ax1, ax2, ax3,
            padding=padding)
    fig.tight_layout()
    return fig, axes
