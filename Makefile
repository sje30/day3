
all: per_rosenblatt_broken.ipynb

%.ipynb: %.qmd
	quarto convert $^

%.pdf: %.qmd
	quarto render $^ --to typst
