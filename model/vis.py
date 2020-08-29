import torch
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt


def tensor2img(t, padding=16):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    mu = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    img = to_pil_image(torch.tensor(t) * std + mu)
    w, h = img.size
    return img.crop((padding, padding, w - padding, h - padding))


def plot_mask_cells(cells):
    fig, (axes) = plt.subplots(1, len(cells), figsize=(12, 5))
    for i, (image, ax) in enumerate(zip(cells, axes)):
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(tensor2img(image), cmap='viridis')
    plt.show()
    return axes


def plot_cells(cells):
    fig, (axes) = plt.subplots(1, len(cells), figsize=(12, 5))
    for i, (image, ax) in enumerate(zip(cells, axes)):
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Sample {i}")
        ax.imshow(image)
    plt.show()
    return axes
