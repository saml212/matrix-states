# measurement-ws build (NeurIPS-2025 kit as the flagged stand-in; see
# venue-requirements.md).
#   make          -> main.pdf (anonymized submission build)
#   make figures  -> regenerate figures/*.pdf from archived raws
#                    (md5-asserted; requires the repo venv's python)
# Build engine: tectonic (brew install tectonic)

all: main.pdf

# Triple pass: deterministic aux fixed point from any starting state
# (house convention, mirrors the sibling EA trees).
main.pdf: main.tex sections/*.tex refs.bib figures/*.pdf
	tectonic --keep-intermediates main.tex
	tectonic --keep-intermediates main.tex
	tectonic --keep-intermediates main.tex

bundle: main.pdf
	mkdir -p bundle/figures
	python3 -c "import re; src=open('main.tex').read(); \
	  flat=re.sub(r'\\\\input\{(sections/[^}]+)\}', lambda m: open(m.group(1)+'.tex').read(), src); \
	  open('bundle/measurement-ws-submission.tex','w').write(flat)"
	cp main.pdf bundle/measurement-ws-submission.pdf
	cp refs.bib neurips.sty bundle/
	cp figures/fig1_lens_contradiction.pdf figures/fig2_tap_localization.pdf bundle/figures/

figures: figures/figure_gen.py
	cd ../.. && .venv/bin/python papers/measurement-ws/figures/figure_gen.py \
	  --out papers/measurement-ws/figures --repo .

clean:
	rm -f main.aux main.log main.bbl main.blg main.out main.toc

.PHONY: all figures clean bundle
