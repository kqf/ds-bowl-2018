import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.morphology import label
from sklearn.pipeline import make_pipeline


class ImageResizer(BaseEstimator, TransformerMixin):
    def __init__(self, height=256, width=256, channels=3):
        super(ImageResizer, self).__init__()
        self.height = height 
        self.width = width
        self.channels = channels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        updated = np.array(
            list(map(lambda x: resize(x, (self.height, self.width, self.channels), mode='constant', preserve_range=True), X))
        )
        return updated

class DataReader(object):

    def __init__(self, datapath="input", stage="stage1", sample=None):
        super(DataReader, self).__init__()
        self.datapath = datapath
        self.stage = stage
        self.sample = sample
        self.folder_structure = {
            "ImageId" : lambda path: path.split('/')[-3],
            "ImageType" : lambda path: path.split('/')[-2],
            "TrainingSplit" : lambda path: path.split('/')[-4].split('_')[1],
            "Stage" : lambda path: path.split('/')[-4].split('_')[0]
        }

    def _dataset(self):
        all_images = glob(os.path.join(self.datapath, '{0}_*'.format(self.stage), '*', '*', '*'))
        imlist = pd.DataFrame({'path': all_images})
        for entry, function in self.folder_structure.items():
            imlist[entry] = imlist['path'].map(function)
        return imlist 

    def read_and_stack(self, images):
        return np.sum(np.stack([imread(c_img) for c_img in images], 0), 0) / 255.0

    def _process_images(self, images):
        pipeline = make_pipeline(
            ImageResizer()
        )
        return pipeline.transform(images)

    def read(self, column="test"):
        dataset = self._dataset()
        output = dataset[dataset.TrainingSplit == column]

        if self.sample:
            output = output.sample(self.sample)

        output = output.rename(index=str, columns={"path": "images"})
        output.images = output.images.apply(lambda x: self.read_and_stack([x]))
        return output[["ImageId"]], self._process_images(output.images)


class TrainDataReader(DataReader):

    def read(self, column="train"):
        dataset = self._dataset()
        output = dataset[dataset.TrainingSplit == column]
        output = pd.merge(
            output[output.ImageType == "images"].rename(index=str, columns={"path": "images"}),
            pd.DataFrame(
                    output[output.ImageType == "masks"].groupby("ImageId")["path"].apply(list)
                ).reset_index().rename(index=str, columns={"path":"masks"}),
                on="ImageId"
        )

        if self.sample:
            output = output.sample(self.sample)

        output.images = output.images.apply(lambda x: self.read_and_stack([x]))
        output.masks = output.masks.apply(self.read_and_stack) 
        return self._process_images(output.images), self._process_images(output.masks)


class RleEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, id_col="ImageId", encoded_col="EncodedPixels"):
        super(RleEncoder, self).__init__()
        self.id_col = id_col
        self.encoded_col = encoded_col

    def _rle_encoding(self, x):
        dots = np.where(x.T.flatten() == 1)[0]
        run_lengths = []
        prev = -2
        for b in dots:
            if (b > prev + 1): 
                run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        return run_lengths

    def _prob_to_rles(self, X, cutoff=0.5):
        lab_img = label(X > cutoff)
        for i in range(1, lab_img.max() + 1):
            yield self._rle_encoding(lab_img == i)

    def transform(self, X):
        test_ids, data = X["ImageId"].values, X["predicted"].values
        new_test_ids, rles = [], []

        for n, id_ in enumerate(test_ids):
            rle = list(self._prob_to_rles(data[n]))
            new_test_ids.extend([id_] * len(rle))
            rles.extend(rle)
            
        output = pd.DataFrame({
            self.id_col: new_test_ids,
            self.encoded_col: rles

        })
        return output


class ResultSaver(BaseEstimator, TransformerMixin):

    def __init__(self, out_columns=["ImageId", "EncodedPixels"], ofile="output.csv"):
        super(ResultSaver, self).__init__()
        self.out_columns = out_columns 
        self.ofile = ofile

    def transform(self, X):
        output = X[self.out_columns]
        output[self.out_columns[-1]] = output[self.out_columns[-1]].apply(lambda x: ' '.join(map(str, x)))
        output.to_csv(self.ofile, index=False)
        return output

class ShaperReporter(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        print("X shape: ", X.shape)
        print("y shape: ", y.shape)
        return self

    def transform(self, X, y=None):
        return X

        
