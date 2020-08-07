competition = data-science-bowl-2018


data:
	kaggle competitions download -c $(competition)
	unzip $(competition).zip -d data/
