# NeurIPS 2026 Workshop (interpretability track) build
#
#   make          -> main.pdf (single-blind, default: real placeholder author block)
#   make anon     -> main-anon.pdf (double-blind build, only if the CFP requires it)
#
# Build engine: tectonic (single binary, auto-fetches packages, no sudo).
# Install: brew install tectonic

all: main.pdf

main.pdf: main.tex sections/*.tex refs.bib
	tectonic --keep-intermediates main.tex

anon: main-anon.tex main.tex sections/*.tex refs.bib
	tectonic --keep-intermediates main-anon.tex

clean:
	rm -f main.aux main.log main.bbl main.blg main.out main.toc
	rm -f main-anon.aux main-anon.log main-anon.bbl main-anon.blg main-anon.out main-anon.toc
	rm -f sections/*.aux

distclean: clean
	rm -f main.pdf main-anon.pdf

.PHONY: all anon clean distclean
