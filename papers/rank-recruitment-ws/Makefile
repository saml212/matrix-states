# NeurReps 2026 EA build ("rank recruitment under teeth")
#   make          -> main.pdf (anonymized review build)
#   make figures  -> regenerate figures/*.pdf from md5-asserted archives
#   make bundle   -> flattened single-file submission in bundle/
# Build engine: tectonic (brew install tectonic). Three passes make the
# jmlr build deterministic from any starting state (aux fixed point; same
# convention as the sibling papers/neurreps-ea/ tree).

all: main.pdf

main.pdf: main.tex sections/*.tex refs.bib figures/*.pdf
	tectonic --keep-intermediates main.tex
	tectonic --keep-intermediates main.tex
	tectonic --keep-intermediates main.tex

# NeurReps CFP asks for a SINGLE .tex file at submission: flatten the
# modular source by inlining every \input{sections/...}.
bundle: main.pdf
	mkdir -p bundle/figures
	python3 -c "import re; src=open('main.tex').read(); \
	  flat=re.sub(r'\\\\input\{(sections/[^}]+)\}', lambda m: open(m.group(1)+'.tex').read(), src); \
	  open('bundle/rank-recruitment-ws-submission.tex','w').write(flat)"
	cp main.pdf bundle/rank-recruitment-ws-submission.pdf
	cp refs.bib jmlr.cls jmlrutils.sty bundle/
	cp figures/fig_forcerank.pdf figures/fig_depth.pdf bundle/figures/

figures: figures/figure_gen.py
	cd ../.. && DRY_RUN_BYPASS=1 .venv/bin/python \
	  papers/rank-recruitment-ws/figures/figure_gen.py \
	  --out papers/rank-recruitment-ws/figures --repo .

clean:
	rm -f main.aux main.log main.bbl main.blg main.out main.toc

.PHONY: all figures clean bundle
