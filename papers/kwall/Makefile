# MOSS @ COLM 2026 build (official COLM 2026 kit, GitHub tag 2026) --
# same vendored kit as papers/reasoning-null-moss/ and papers/measurement-ws/.
#   make          -> main.pdf (anonymized submission build; [submission] option)
#   make bundle   -> flattened single-file submission in bundle/
# Build engine: tectonic (brew install tectonic). Three passes for a
# deterministic build from any starting state (house convention).

all: main.pdf

main.pdf: main.tex sections/*.tex refs.bib
	tectonic --keep-intermediates main.tex
	tectonic --keep-intermediates main.tex
	tectonic --keep-intermediates main.tex

bundle: main.pdf
	mkdir -p bundle
	python3 -c "import re; src=open('main.tex').read(); \
	  flat=re.sub(r'\\\\input\{(sections/[^}]+)\}', lambda m: open(m.group(1)+'.tex').read(), src); \
	  open('bundle/kwall-submission.tex','w').write(flat)"
	cp main.pdf bundle/kwall-submission.pdf
	cp refs.bib colm2026_conference.sty colm2026_conference.bst fancyhdr.sty natbib.sty math_commands.tex bundle/ 2>/dev/null || true

clean:
	rm -f main.aux main.log main.bbl main.blg main.out main.toc

.PHONY: all clean bundle
