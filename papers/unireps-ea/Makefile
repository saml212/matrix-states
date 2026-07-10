# UniReps 2026 EA build (house scaffold pending official style file)
#   make          -> main.pdf (anonymized submission build; style default)
#   make figures  -> regenerate figures/*.pdf from archived experiment JSONs
#                    (md5-asserted; requires the repo venv's python)
# Build engine: tectonic (brew install tectonic)

all: main.pdf

# NOTE: triple pass mirrors the neurreps-ea S-A guard (aux fixed-point
# transient there, verified 2026-07-10). No bistability observed in THIS
# tree, but the repeated pass is harmless and makes the build
# deterministic from any starting aux state.
main.pdf: main.tex sections/*.tex refs.bib figures/*.pdf
	tectonic --keep-intermediates main.tex
	tectonic --keep-intermediates main.tex
	tectonic --keep-intermediates main.tex

bundle: main.pdf
	mkdir -p bundle/figures
	python3 -c "import re; src=open('main.tex').read(); \
	  flat=re.sub(r'\\\\input\{(sections/[^}]+)\}', lambda m: open(m.group(1)+'.tex').read(), src); \
	  open('bundle/unireps-ea-submission.tex','w').write(flat)"
	cp main.pdf bundle/unireps-ea-submission.pdf
	cp refs.bib neurips.sty bundle/
	cp figures/fig1_convergence.pdf figures/fig2_razor_step.pdf bundle/figures/

figures: figures/figure_gen.py
	cd ../.. && .venv/bin/python papers/unireps-ea/figures/figure_gen.py \
	  --out papers/unireps-ea/figures --repo .

clean:
	rm -f main.aux main.log main.bbl main.blg main.out main.toc

.PHONY: all figures clean bundle
