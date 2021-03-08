# Copyright 2020, Abhirup Ghosh <abhirup.ghosh.184098@gmail.com>, All rights reserved.
PAPER=pa_tgr_section

.SUFFIXES:
.SUFFIXES: .bib .pdf .tex
.PHONY: clean

run: $(PAPER).pdf

view: $(PAPER).pdf
	open -a Preview.app $(PAPER).pdf

$(PAPER).pdf: $(PAPER).bbl $(PAPER).tex
	pdflatex $(PAPER).tex -draftmode
	pdflatex $(PAPER).tex 

$(PAPER).bbl: $(PAPER).aux
	bibtex $(PAPER).aux

$(PAPER).aux: $(PAPER).bib
	pdflatex $(PAPER).tex -draftmode
	pdflatex $(PAPER).tex -draftmode

clean:
	rm -rf *.aux *.lof *.log *.lot *.toc *.bbl *.blg *pdf
