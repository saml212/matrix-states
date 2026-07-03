# ICML MI Workshop 2026 build
#
# Two builds from one source:
#   make anon    -> paper-anon.pdf (workshop submission, double-blind)
#   make pub     -> paper.pdf      (arXiv + share-with-friends, real authors)
#   make         -> both
#
# Build engine: tectonic (single binary, auto-fetches packages, no sudo).
# Install: brew install tectonic
#
# Generate figures first (requires matplotlib): make figures

all: paper-anon.pdf paper.pdf

paper-anon.pdf: paper-anon.tex main.tex sections/*.tex refs.bib figures/*.pdf
	tectonic --keep-intermediates paper-anon.tex

paper.pdf: paper.tex main.tex sections/*.tex refs.bib figures/*.pdf
	tectonic --keep-intermediates paper.tex

anon: paper-anon.pdf
pub: paper.pdf

figures:
	cd figures && python3 generate_figures.py

clean:
	rm -f paper.aux paper.log paper.bbl paper.blg paper.out paper.toc
	rm -f paper-anon.aux paper-anon.log paper-anon.bbl paper-anon.blg paper-anon.out paper-anon.toc
	rm -f sections/*.aux

distclean: clean
	rm -f paper.pdf paper-anon.pdf

.PHONY: all anon pub figures clean distclean
