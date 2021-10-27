.PHONY: tex all spellcheck

all: tex clean

tex:
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make BachelorThesis.tex

clean:
	latexmk -c

spellcheck:
	find . -name "*.tex" -exec aspell --lang=en --mode=tex check "{}" \;