competition = data-science-bowl-2018


data/cells: data
	mkdir -p data/cells
	unzip data/stage1_train.zip -d data/cells

data:
	kaggle competitions download -c $(competition) -p data
	unzip data/$(competition).zip -d data/
