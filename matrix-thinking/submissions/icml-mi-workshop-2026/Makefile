# ICML MI Workshop 2026 build
#
# Requirements: pdflatex, bibtex (texlive-latex-base, texlive-bibtex-extra).
# Generate figures first (requires matplotlib):
#
#   make figures
#
# Then build the PDF:
#
#   make
#
# The paper uses a stand-in preamble. Before submission, download
# icml2026.sty from https://icml.cc/Conferences/2026/CallForPapers
# and swap it into main.tex (see comment block at top of main.tex).

TEX = main

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex sections/*.tex refs.bib figures/*.pdf
	pdflatex $(TEX).tex
	bibtex $(TEX)
	pdflatex $(TEX).tex
	pdflatex $(TEX).tex

figures:
	cd figures && python3 generate_figures.py

clean:
	rm -f $(TEX).aux $(TEX).log $(TEX).bbl $(TEX).blg $(TEX).out $(TEX).toc
	rm -f sections/*.aux

distclean: clean
	rm -f $(TEX).pdf

.PHONY: all figures clean distclean
