competition = data-science-bowl-2018


train: data/cells/merged.txt
	echo "Training"

data/cells/merged.txt: data/cells
	merge-masks --path data/cells
	echo "The dataset has been extracted" > data/cells/merged.txt

data/cells: data
	mkdir -p data/cells
	unzip -qq data/stage1_train.zip -d data/cells

data:
	kaggle competitions download -c $(competition) -p data
	unzip -qq data/$(competition).zip -d data/
