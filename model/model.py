import torch
import skorch
from torchvision import models


def make_decoder_block(in_channels, middle_channels, out_channels):
    block = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, middle_channels, 3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.ConvTranspose2d(
            middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(inplace=True))
    return block


class UNet(torch.nn.Module):
    """UNet Model inspired by the the original UNet paper
    Parameters
    ----------
    pretrained: bool (default=True)
        Option to use pretrained vgg16_bn based on ImageNet
    References
    ----------
    .. [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox, 2015,
        "U-Net: Convolutional Networks for Biomedical Image Segmentation,".
        "MICCAI" `<https://arxiv.org/abs/1505.04597>`_
    """

    def __init__(self, pretrained=False):
        super().__init__()
        encoder = models.vgg16_bn(pretrained=pretrained).features
        self.conv1 = encoder[:6]
        self.conv2 = encoder[6:13]
        self.conv3 = encoder[13:23]
        self.conv4 = encoder[23:33]
        self.conv5 = encoder[33:43]

        self.center = torch.nn.Sequential(
            encoder[43],  # MaxPool
            make_decoder_block(512, 512, 256)
        )

        self.dec5 = make_decoder_block(256 + 512, 512, 256)
        self.dec4 = make_decoder_block(256 + 512, 512, 256)
        self.dec3 = make_decoder_block(256 + 256, 256, 64)
        self.dec2 = make_decoder_block(64 + 128, 128, 32)
        self.dec1 = torch.nn.Sequential(
            torch.nn.Conv2d(32 + 64, 32, 3, padding=1),
            torch.nn.ReLU(inplace=True)
        )
        self.final = torch.nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(conv5)

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        return self.final(dec1)


def build_model():
    model = skorch.NeuralNet(
        UNet,
        criterion__padding=16,
        batch_size=32,
        max_epochs=20,
        optimizer__momentum=0.9,
        iterator_train__shuffle=True,
        iterator_train__num_workers=4,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=4,
        callbacks=[
            skorch.callbacks.Checkpoint(f_params='best-params.pt'),
        ],
        device='cuda',
    )

    return model