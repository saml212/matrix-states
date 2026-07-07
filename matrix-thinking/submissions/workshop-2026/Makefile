# Workshop-cut (capacity trilogy) build
#
#   make          -> main.pdf (single-blind, default: real placeholder author block)
#   make figures  -> regenerate figures/fig_cliff.* from ../iclr-2027/figures/
#                    (fig_cliff is authored there; copy, don't re-derive) and
#                    figures/fig_dose.* from ../iclr-2027/figures/make_fig_dose.py
#                    (requires python3 + numpy + scipy + matplotlib; NOT a
#                    dependency of `make` -- the figure PDFs are committed
#                    build artifacts, like main.bbl)
#
# Build engine: tectonic (single binary, auto-fetches packages, no sudo).
# Install: brew install tectonic

ICLR_FIGDIR = ../iclr-2027/figures

all: main.pdf

main.pdf: main.tex sections/*.tex refs.bib figures/*.pdf
	tectonic --keep-intermediates main.tex

figures:
	python3 $(ICLR_FIGDIR)/make_fig_dose.py
	cp $(ICLR_FIGDIR)/fig_dose.pdf $(ICLR_FIGDIR)/fig_dose.png figures/
	cp $(ICLR_FIGDIR)/fig_cliff.pdf $(ICLR_FIGDIR)/fig_cliff.png figures/

clean:
	rm -f main.aux main.log main.bbl main.blg main.out main.toc
	rm -f sections/*.aux

distclean: clean
	rm -f main.pdf

.PHONY: all figures clean distclean
