import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os
from skimage.io import imread
import matplotlib.pyplot as plt
import seaborn as sns



class DataManager(object):
    def __init__(self, datapath="input", stage="stage1"):
        super(DataManager, self).__init__()
        self.datapath = datapath
        self.stage = stage
        self._imagelist = None

    @property
    def imagelist(self):
        if self._imagelist is not None:
            return self._imagelist
        self._imagelist = self.load_images()
        return self._imagelist

    def list_of_images(self):
        all_images = glob(os.path.join(self.datapath, '{0}_*'.format(self.stage), '*', '*', '*'))
        imlist = pd.DataFrame({'path': all_images})
        img_id = lambda in_path: in_path.split('/')[-3]
        img_type = lambda in_path: in_path.split('/')[-2]
        img_group = lambda in_path: in_path.split('/')[-4].split('_')[1]
        img_stage = lambda in_path: in_path.split('/')[-4].split('_')[0]
        imlist['ImageId'] = imlist['path'].map(img_id)
        imlist['ImageType'] = imlist['path'].map(img_type)
        imlist['TrainingSplit'] = imlist['path'].map(img_group)
        imlist['Stage'] = imlist['path'].map(img_stage)
        return imlist 


    def labels(datapath="input", stage="stage1"):
        train_labels = pd.read_csv(
            os.path.join(self.datapath,'{}_train_labels.csv'.format(self.stage))
        )

        train_labels['EncodedPixels'] = train_labels['EncodedPixels'].map(
            lambda ep: [int(x) for x in ep.split(' ')]
        )
        return train_labels


    def load_images(self):
        imlist = self.list_of_images()
        output = pd.merge(
            imlist[imlist.ImageType == "images"].rename(index=str, columns={"path": "image"}),
            pd.DataFrame(
                    imlist[imlist.ImageType == "masks"].groupby("ImageId")["path"].apply(list)
                ).reset_index().rename(index=str, columns={"path":"masks"}),
                on="ImageId"
        )

        print('here')
        output = output.sample(10)
        def read_and_stack(images):
            return np.sum(np.stack([imread(c_img) for c_img in images], 0), 0) / 255.0

        output.image = output.image.apply(lambda x: read_and_stack([x]))
        output.masks = output.masks.apply(read_and_stack) 
        return output

    def images(self):
        ofilename = "{0}/{1}-images.npy".format(self.datapath, self.stage)
        try: 
            return np.load(ofilename)
        except FileNotFoundError:
            imlist = self.imagelist.image.values
        # np.save(ofilename, imlist)
        return imlist

    def masks(self):
        ofilename = "{0}/{1}-masks".format(self.datapath, self.stage)
        try: 
            return np.load(ofilename)
        except FileNotFoundError:
            masklist = self.imagelist.masks.values
        # np.save(ofilename, masklist)
        return masklist 
