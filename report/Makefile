# You want latexmk to *always* run, because make does not have all the info.
# Also, include non-file targets in .PHONY so they are run regardless of any
# file of the given name existing.
.PHONY: main.pdf all clean

all: main.pdf

main.pdf: main.tex
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make main.tex

clean:
	latexmk -CA
