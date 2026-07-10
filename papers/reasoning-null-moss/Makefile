# MOSS @ COLM 2026 build (official COLM 2026 kit, GitHub tag 2026)
#   make          -> main.pdf (anonymized submission build; [submission] option)
#   make figures  -> regenerate figures/*.pdf from archived experiment JSONs
#                    (md5-asserted; requires the repo venv's python)
# Build engine: tectonic (brew install tectonic). Three passes for a
# deterministic build from any starting state (house convention).

all: main.pdf

main.pdf: main.tex sections/*.tex refs.bib figures/*.pdf
	tectonic --keep-intermediates main.tex
	tectonic --keep-intermediates main.tex
	tectonic --keep-intermediates main.tex

bundle: main.pdf
	mkdir -p bundle
	python3 -c "import re; src=open('main.tex').read(); \
	  flat=re.sub(r'\\\\input\{(sections/[^}]+)\}', lambda m: open(m.group(1)+'.tex').read(), src); \
	  open('bundle/reasoning-null-moss-submission.tex','w').write(flat)"
	cp main.pdf bundle/reasoning-null-moss-submission.pdf
	cp refs.bib colm2026_conference.sty colm2026_conference.bst fancyhdr.sty natbib.sty math_commands.tex bundle/ 2>/dev/null || true
	mkdir -p bundle/figures
	cp figures/fig1_validity_gates.pdf figures/fig2_dissociation.pdf figures/fig3_transient_replication.pdf bundle/figures/

figures: figures/figure-gen.py
	cd ../.. && .venv/bin/python papers/reasoning-null-moss/figures/figure-gen.py

clean:
	rm -f main.aux main.log main.bbl main.blg main.out main.toc

.PHONY: all figures clean bundle
