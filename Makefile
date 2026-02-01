all: main.pdf

main.pdf: main.tex refs.bib
	pdflatex -shell-escape main
	bibtex main
	pdflatex -shell-escape main
	pdflatex -shell-escape main

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz *.bbl *.blg *.pyg
	rm -rf _minted-main
	rm -f main.pdf

.PHONY: all clean
