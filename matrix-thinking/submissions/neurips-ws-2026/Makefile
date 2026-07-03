# NeurIPS 2026 Workshop (interpretability track) build
#
#   make          -> main.pdf (single-blind, default: real placeholder author block)
#   make anon     -> main-anon.pdf (double-blind build, only if the CFP requires it)
#   make figures  -> regenerate figures/*.pdf from archived experiment JSONs
#                    (requires python3 + numpy + matplotlib; NOT a dependency
#                    of `make`/`make anon` -- the figure PDFs are committed
#                    build artifacts, like main.bbl, so a plain LaTeX-only
#                    environment can still build the paper)
#
# Build engine: tectonic (single binary, auto-fetches packages, no sudo).
# Install: brew install tectonic

all: main.pdf

main.pdf: main.tex sections/*.tex refs.bib figures/*.pdf
	tectonic --keep-intermediates main.tex

anon: main-anon.tex main.tex sections/*.tex refs.bib figures/*.pdf
	tectonic --keep-intermediates main-anon.tex

figures: figures/make_figures.py
	python3 figures/make_figures.py

clean:
	rm -f main.aux main.log main.bbl main.blg main.out main.toc
	rm -f main-anon.aux main-anon.log main-anon.bbl main-anon.blg main-anon.out main-anon.toc
	rm -f sections/*.aux

distclean: clean
	rm -f main.pdf main-anon.pdf

.PHONY: all anon figures clean distclean
