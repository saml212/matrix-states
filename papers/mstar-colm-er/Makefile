# COLM 2026 Efficient Reasoning workshop submission build.
#   make          -> main.pdf (anonymized submission build; [submission] option)
#   make figures  -> regenerate figures/*.pdf + figures/tables_generated.tex
#                    from archived experiment JSONs (md5-asserted; repo venv)
#   make bundle   -> flattened single-tex submission bundle
# Build engine: tectonic (three passes for a deterministic fixed point,
# the convention the sibling paper trees in papers/ already use).

all: main.pdf

main.pdf: main.tex sections/*.tex refs.bib figures/*.pdf figures/tables_generated.tex
	tectonic --keep-intermediates main.tex
	tectonic --keep-intermediates main.tex
	tectonic --keep-intermediates main.tex

figures: figures/figure-gen.py
	cd ../.. && .venv/bin/python papers/mstar-colm-er/figures/figure-gen.py \
	  --repo . --out papers/mstar-colm-er/figures

bundle: main.pdf
	mkdir -p bundle
	python3 -c "import re; src=open('main.tex').read(); \
	  flat=re.sub(r'\\\\input\{(sections/[^}]+)\}', lambda m: open(m.group(1)+'.tex').read(), src); \
	  open('bundle/mstar-colm-er-submission.tex','w').write(flat)"
	cp main.pdf bundle/mstar-colm-er-submission.pdf
	cp refs.bib colm2026_conference.sty colm2026_conference.bst \
	  math_commands.tex fancyhdr.sty natbib.sty bundle/
	mkdir -p bundle/figures
	cp figures/fig1_horizon.pdf figures/fig2_szero.pdf \
	  figures/tables_generated.tex bundle/figures/

clean:
	rm -f main.aux main.log main.bbl main.blg main.out main.toc

.PHONY: all figures clean bundle
