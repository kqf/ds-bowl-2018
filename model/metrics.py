import numpy as np


def iou_approx(true_masks, probas, padding=0):
    _, sx, sy = probas.shape
    true_masks = true_masks[:, padding:sx - padding, padding:sy - padding]
    probas = probas[:, padding:sx - padding, padding:sy - padding]
    preds = 1 / (1 + np.exp(-probas))

    approx_intersect = np.sum(np.minimum(preds, true_masks), axis=(1, 2))
    approx_union = np.sum(np.maximum(preds, true_masks), axis=(1, 2))
    return np.mean(approx_intersect / approx_union)


def iou(predictions, labels, verbose=False):
    return np.array([
        _iou_single(p, l, verbose) for p, l in zip(predictions, labels)
    ]).mean()


def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(
        false_positives), np.sum(false_negatives)
    return tp, fp, fn


def _iou_single(y_pred, labels, verbose):
    # Compute number of objects
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    if verbose:
        print("Number of true objects:", true_objects)
        print("Number of predicted objects:", pred_objects)

    # Compute intersection between all objects
    intersection = np.histogram2d(
        labels.flatten(),
        y_pred.flatten(),
        bins=(true_objects, pred_objects)
    )[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Loop over IoU thresholds
    prec = []
    if verbose:
        print("Thresh\tTP\tFP\tFN\tPrec.")

    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp / (tp + fp + fn)
        if verbose:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if verbose:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

    return np.mean(prec)
