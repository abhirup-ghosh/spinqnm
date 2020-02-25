DOCS = TGR_QNM_pSEOBNRv4HM.pdf
BIB = references.bib
BYPROD = *.aux *.log *.out *.dvi *.blg *.bbl *Notes.bib
TEXFILES = TGR_QNM_pSEOBNRv4HM.tex
all: pdf
default: all

pdf: $(BIB) $(TEXFILES)
	pdflatex TGR_QNM_pSEOBNRv4HM
	bibtex TGR_QNM_pSEOBNRv4HM
	pdflatex TGR_QNM_pSEOBNRv4HM
	pdflatex TGR_QNM_pSEOBNRv4HM

clean:
	rm -rf $(BYPROD) $(DOCS)

