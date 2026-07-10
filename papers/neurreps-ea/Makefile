# NeurReps 2026 EA build (house scaffold pending official style file)
#   make          -> main.pdf (single-blind placeholder-author build)
#   make anon     -> main-anon.pdf (double-blind build)
#   make figures  -> regenerate figures/*.pdf from archived experiment JSONs
#                    (md5-asserted; requires the repo venv's python)
# Build engine: tectonic (brew install tectonic)

all: main.pdf

main.pdf: main.tex sections/*.tex refs.bib figures/*.pdf
	tectonic --keep-intermediates main.tex

anon: main-anon.tex main.tex sections/*.tex refs.bib figures/*.pdf
	tectonic --keep-intermediates main-anon.tex

figures: figures/figure_gen.py
	cd ../.. && .venv/bin/python papers/neurreps-ea/figures/figure_gen.py \
	  --out papers/neurreps-ea/figures --repo .

clean:
	rm -f main.aux main.log main.bbl main.blg main.out main.toc
	rm -f main-anon.aux main-anon.log main-anon.bbl main-anon.blg main-anon.out main-anon.toc

.PHONY: all anon figures clean
