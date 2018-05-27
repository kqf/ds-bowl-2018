import unittest

import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import skimage.segmentation

from model.metrics import iou


class TestIoUMetrics(unittest.TestCase):

    def test_calculates_the_metrics(self):
        # Load a single image and its associated masks
        imid = '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9'
        imfile = "input/stage1_train/{}/images/{}.png".format(imid, imid)
        masks = "input/stage1_train/{}/masks/*.png".format(imid)

        image = skimage.io.imread(imfile)
        masks = skimage.io.imread_collection(masks).concatenate()
        height, width, _ = image.shape
        num_masks = masks.shape[0]

        # Make a ground truth label image (pixel value is index of object label)
        labels = np.zeros((height, width), np.uint16)
        for index in range(0, num_masks):
            labels[masks[index] > 0] = index + 1

        # Simulate an imperfect submission
        offset = 2  # offset pixels
        y_pred = labels[offset:, offset:]
        y_pred = np.pad(y_pred, ((0, offset), (0, offset)), mode="constant")
        y_pred[y_pred == 20] = 0  # Remove one object
        y_pred, _, _ = skimage.segmentation.relabel_sequential(
            y_pred)  # Relabel objects

        # Show simulated predictions
        fig = plt.figure()
        plt.imshow(y_pred)
        plt.title("Simulated imperfect submission")
        print("The final metics", iou([y_pred], [labels], verbose=True))
