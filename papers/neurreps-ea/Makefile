# NeurReps 2026 EA build (house scaffold pending official style file)
#   make          -> main.pdf (anonymized submission build; style default)
#   make figures  -> regenerate figures/*.pdf from archived experiment JSONs
#                    (md5-asserted; requires the repo venv's python)
# Build engine: tectonic (brew install tectonic)

all: main.pdf

# NOTE (round-2 v2 audit S-A, verified 2026-07-10): a clean-tree build
# transits a mis-paginated 12pp layout (aux fixed-point transient; blank
# stretches in the bibliography) and converges to the correct dense 8pp
# layout on the THIRD tectonic pass; incremental rebuilds are stable.
# Three passes make the build deterministic from any starting state.
main.pdf: main.tex sections/*.tex refs.bib figures/*.pdf
	tectonic --keep-intermediates main.tex
	tectonic --keep-intermediates main.tex
	tectonic --keep-intermediates main.tex

# NeurReps CFP asks for a SINGLE .tex file at submission: flatten the
# modular source by inlining every \input{sections/...}.
bundle: main.pdf
	mkdir -p bundle
	python3 -c "import re; src=open('main.tex').read(); \
	  flat=re.sub(r'\\\\input\{(sections/[^}]+)\}', lambda m: open(m.group(1)+'.tex').read(), src); \
	  open('bundle/neurreps-ea-submission.tex','w').write(flat)"
	cp main.pdf bundle/neurreps-ea-submission.pdf
	cp refs.bib jmlr.cls jmlrutils.sty bundle/
	mkdir -p bundle/figures
	cp figures/fig1_razor_step.pdf figures/fig2_rank_tracking.pdf bundle/figures/

figures: figures/figure_gen.py
	cd ../.. && .venv/bin/python papers/neurreps-ea/figures/figure_gen.py \
	  --out papers/neurreps-ea/figures --repo .

clean:
	rm -f main.aux main.log main.bbl main.blg main.out main.toc

.PHONY: all figures clean bundle
