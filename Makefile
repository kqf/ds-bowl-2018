competition = data-science-bowl-2018


train: data/cells/merged.txt
	nuclei train --path data/cells/


dataset: data/cells/merged.txt
	echo "Successfully loaded"

data/cells/merged.txt: data/cells
	merge-masks --path data/cells
	echo "The dataset has been extracted" > data/cells/merged.txt

data/cells: data/stage1_train.zip
	mkdir -p data/cells
	unzip -qq data/stage1_train.zip -d data/cells

data/stage1_train.zip:
	kaggle competitions download -c $(competition) -p data
	unzip -qq data/$(competition).zip -d data/


.PHONY: dataset
