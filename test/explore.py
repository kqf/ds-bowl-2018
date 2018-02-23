import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os
from skimage.io import imread
import matplotlib.pyplot as plt
import seaborn as sns

dsb_data_dir = os.path.join('input')
stage_label = 'stage1'

def load_images():
	all_images = glob(os.path.join(dsb_data_dir, 'stage1_*', '*', '*', '*'))
	img_df = pd.DataFrame({'path': all_images})
	img_id = lambda in_path: in_path.split('/')[-3]
	img_type = lambda in_path: in_path.split('/')[-2]
	img_group = lambda in_path: in_path.split('/')[-4].split('_')[1]
	img_stage = lambda in_path: in_path.split('/')[-4].split('_')[0]
	img_df['ImageId'] = img_df['path'].map(img_id)
	img_df['ImageType'] = img_df['path'].map(img_type)
	img_df['TrainingSplit'] = img_df['path'].map(img_group)
	img_df['Stage'] = img_df['path'].map(img_stage)
	return img_df

def load_labels():
	train_labels = pd.read_csv(os.path.join(dsb_data_dir,'{}_train_labels.csv'.format(stage_label)))
	train_labels['EncodedPixels'] = train_labels['EncodedPixels'].map(
		lambda ep: [int(x) for x in ep.split(' ')]
	)
	return train_labels

def train(img_df):
	train_df = img_df.query('TrainingSplit=="train"')
	train_rows = []
	group_cols = ['Stage', 'ImageId']
	for n_group, n_rows in train_df.groupby(group_cols):
	    c_row = {col_name: col_value for col_name, col_value in zip(group_cols, n_group)}
	    c_row['masks'] = n_rows.query('ImageType == "masks"')['path'].values.tolist()
	    c_row['images'] = n_rows.query('ImageType == "images"')['path'].values.tolist()
	    train_rows += [c_row]
	train_img_df = pd.DataFrame(train_rows)    
	IMG_CHANNELS = 3
	def read_and_stack(in_img_list):
	    return np.sum(np.stack([imread(c_img) for c_img in in_img_list], 0), 0)/255.0
	train_img_df['images'] = train_img_df['images'].map(read_and_stack).map(lambda x: x[:,:,:IMG_CHANNELS])
	train_img_df['masks'] = train_img_df['masks'].map(read_and_stack).map(lambda x: x.astype(int))
	train_img_df.sample(1)



def main():
	train_labels = load_labels()
	print("Labels:")
	print(train_labels.sample(3))

	print()
	print("Images:")
	img_df = load_images()
	print(img_df.sample(2))


if __name__ == '__main__':
	main()