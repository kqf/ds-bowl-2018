import torch
from torchvision.transforms.functional import to_pil_image


def tensor2pil(t, padding=16):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    mu = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    img = to_pil_image(t.mul(std).add_(mu))
    w, h = img.size
    return img.crop((padding, padding, w - padding, h - padding))
