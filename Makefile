all: main.pdf

main.pdf: main.tex refs.bib
	pdflatex -shell-escape main
	bibtex main
	pdflatex -shell-escape main
	pdflatex -shell-escape main
	pdflatex -shell-escape main

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz *.bbl *.blg *.pyg
	rm -rf _minted-main
	rm -f main.pdf



latexmk:
	latexmk -pdf -shell-escape main.tex

latexmk-clean:
	latexmk -C
	rm -rf _minted-main

.PHONY: all clean latexmk latexmk-clean

