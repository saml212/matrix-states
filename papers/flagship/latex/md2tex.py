#!/usr/bin/env python3
"""md2tex.py - deterministic converter: sections/*.md -> latex/sections/*.tex

The markdown sections are the prose source of record (evidence comments
included). This script regenerates the .tex inputs from them so detector/
gauntlet prose edits never have to be applied twice. Rerunnable; output is
overwritten every run.

Conversions:
- '# <num> Title' / '# Appendix X Title' -> \\section{Title}
- '## N.M Title'                          -> \\subsection{Title}
- HTML evidence comments                  -> '% evidence: ...' comments
- '**Figure N caption.** text'            -> figure environment (file map below)
- markdown tables                         -> booktabs tabular (Table 1 gets its caption)
- author-year literals                    -> natbib macros (map below)
- `code`                                  -> \\texttt{}
- **bold** / *ital*                       -> \\textbf{} / \\emph{}
"""
import re, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'sections')
DST = os.path.join(HERE, 'sections')

# width fraction chosen from each PDF's natural size (238pt-wide panels
# sit at ~0.62\linewidth; 490pt-wide multi-panel rows take the full line)
FIGMAP = {
    '1': ('fig1_rank_vs_dmin.pdf', 'fig:rank-vs-dmin', 0.62),
    '2': ('fig2_forcerank_staircase.pdf', 'fig:forcerank', 1.0),
    '3': ('fig3_recall_separation.pdf', 'fig:recall', 1.0),
    '4': ('fig4_tap_localization.pdf', 'fig:taps', 1.0),
    '5': ('fig5_attractor_ladder.pdf', 'fig:ladder', 0.62),
    '6': ('fig6_2x2_mitigations.pdf', 'fig:mitigations', 0.62),
    '7': ('fig7_fixscale_transfer.pdf', 'fig:fixscale', 0.62),
    'A1': ('figA1_complement_scaffold.pdf', 'fig:complement', 1.0),
}

# Order matters: longer literals first.
CITEMAP = [
    ('Nichani, Lee, and\nBietti (2024)', r'\citet{nichani2024understandingfactualrecalltransformers}'),
    ('Nichani, Lee, and Bietti (2024)', r'\citet{nichani2024understandingfactualrecalltransformers}'),
    ('(Nichani et al., 2024)', r'\citep{nichani2024understandingfactualrecalltransformers}'),
    ('Arora et\nal. (2023)', r'\citet{arora2023zoologymeasuringimprovingrecall}'),
    ('Arora et al. (2023)', r'\citet{arora2023zoologymeasuringimprovingrecall}'),
    ('Jelassi et al. (2024)', r'\citet{jelassi2024repeatmetransformersbetter}'),
    ('Olsson et al.\n(2022)', r'\citet{olsson2022incontextlearninginductionheads}'),
    ('Olsson et al. (2022)', r'\citet{olsson2022incontextlearninginductionheads}'),
    ('(Olsson et al.,\n2022)', r'\citep{olsson2022incontextlearninginductionheads}'),
    ('(Olsson et al., 2022)', r'\citep{olsson2022incontextlearninginductionheads}'),
    ('(Xiao et al., 2024)', r'\citep{xiao2024efficientstreaminglanguagemodels}'),
    ('DeltaNet and Gated DeltaNet (Yang et\nal.)',
     r'DeltaNet \citep{yang2025parallelizinglineartransformersdelta} and Gated DeltaNet \citep{yang2025gateddeltanetworksimproving}'),
    ('DeltaNet and Gated DeltaNet (Yang et al.)',
     r'DeltaNet \citep{yang2025parallelizinglineartransformersdelta} and Gated DeltaNet \citep{yang2025gateddeltanetworksimproving}'),
    ('The delta-rule family (DeltaNet and its gated descendants)',
     r'The delta-rule family (DeltaNet and its gated descendants; \citealp{yang2025parallelizinglineartransformersdelta,yang2025gateddeltanetworksimproving})'),
    ('(arXiv:2602.04852; arXiv:2602.02195)',
     r'\citep{nazari2026keystatereductionlinear,sun2026staterankdynamicslinear}'),
    ('(Kimi\nLinear, Section 4 of arXiv:2510.26692; Qwen3-Next)',
     r'(Kimi Linear, \citealp{kimiteam2025kimilinearexpressiveefficient}; Qwen3-Next)'),
    ('(Kimi Linear, Section 4 of arXiv:2510.26692; Qwen3-Next)',
     r'(Kimi Linear, \citealp{kimiteam2025kimilinearexpressiveefficient}; Qwen3-Next)'),
]

def esc_text(s):
    # escape LaTeX specials in non-math text; leave $...$ spans alone
    out, parts = [], re.split(r'(\$[^$]*\$)', s)
    for i, p in enumerate(parts):
        if i % 2 == 1:
            out.append(p)
        else:
            p = p.replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')
            out.append(p)
    return ''.join(out)

def inline(s):
    s = re.sub(r'`([^`]+)`', lambda m: r'\texttt{' + m.group(1).replace('_', r'\_') + '}', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', s)
    return s

def convert(md, is_appendix):
    for lit, mac in CITEMAP:
        md = md.replace(lit, mac)
    # evidence comments: STRIP from the .tex output (a mid-line % comment
    # would eat the rest of its line; render-inspection C1 showed escaping
    # them prints them as body text). Traceability lives in the .md source.
    md = re.sub(r'\s*<!--.*?-->', '', md, flags=re.S)
    # bold spans line breaks inside md list items; convert on the whole
    # text before line-splitting so no literal ** survives (render S2)
    md = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', md)
    # single-star italics (render v2): whole-text, math-safe — the content
    # class excludes $ so asterisks inside two different math spans can
    # never pair up across the text between them
    md = re.sub(r'(?<![\w$*])\*([A-Za-z][^*$]{0,80}?)\*(?![\w*])',
                r'\\emph{\1}', md, flags=re.S)

    lines, out, i = md.split('\n'), [], 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r'^# (?:Appendix\s+\w+\s+)?(?:\d+\s+)?(.*)', ln)
        if ln.startswith('# ') and m:
            title = m.group(1) if m.group(1) else ln[2:]
            out.append(r'\section{' + esc_text(title) + '}')
            i += 1; continue
        m = re.match(r'^## (?:\d+\.\d+\s+|[A-Z]\.\d+\s+)?(.*)', ln)
        if ln.startswith('## ') and m:
            out.append(r'\subsection{' + esc_text(m.group(1)) + '}')
            i += 1; continue
        # figure caption blocks
        m = re.match(r'^\\textbf\{Figure (\w+) caption\.\}\s*(.*)', inline(ln))
        if m:
            fignum, first = m.group(1), m.group(2)
            cap = [first]
            i += 1
            while i < len(lines) and lines[i].strip() != '':
                cap.append(lines[i]); i += 1
            fname, label, wfrac = FIGMAP[fignum]
            body = esc_text(inline(' '.join(cap)))
            width = r'\linewidth' if wfrac >= 1.0 else str(wfrac) + r'\linewidth'
            out += [r'\begin{figure}[t]', r'\centering',
                    r'\includegraphics[width=' + width + ']{' + fname + '}',
                    r'\caption{' + body + '}', r'\label{' + label + '}',
                    r'\end{figure}', '']
            continue
        # markdown tables
        if ln.strip().startswith('|') and i + 1 < len(lines) and re.match(r'^\s*\|[\s\-|]+\|\s*$', lines[i + 1]):
            hdr = [c.strip() for c in ln.strip().strip('|').split('|')]
            rows, j = [], i + 2
            while j < len(lines) and lines[j].strip().startswith('|'):
                rows.append([c.strip() for c in lines[j].strip().strip('|').split('|')])
                j += 1
            ncol = len(hdr)
            is_seed_table = hdr and hdr[0] == 'arm'
            if is_seed_table:
                colspec = 'l' + 'c' * (ncol - 1)
                cell = lambda c: esc_text(inline(c))
                env = [r'\begin{table}[t]', r'\centering',
                       r'\caption{Episodic recall $\mathrm{acc}_A$ per seed and arm at matched parameters, tokens, and compute; chance $1/32=0.03125$, demonstration bar $0.09375$.}',
                       r'\label{tab:recall}',
                       r'\begin{tabular}{' + colspec + '}', r'\toprule',
                       ' & '.join(cell(h) for h in hdr) + r' \\', r'\midrule']
                for r in rows:
                    env.append(' & '.join(cell(c) for c in r) + r' \\')
                env += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
            else:
                # long-prose config/manifest tables: ragged p-columns, small font,
                # \path{} for code tokens so long identifiers break at punctuation
                widths = {2: [0.16, 0.76], 3: [0.10, 0.52, 0.30]}.get(
                    ncol, [round(0.92 / ncol, 2)] * ncol)
                colspec = ''.join(r'>{\raggedright\arraybackslash}p{%.2f\linewidth}' % w
                                  for w in widths)
                def cell(c):
                    c = re.sub(r'`([^`]+)`', lambda m: r'\path{' + m.group(1) + '}', c)
                    c = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', c)
                    return esc_text(c)
                env = ['{\\scriptsize',
                       r'\begin{tabular}{' + colspec + '}', r'\toprule',
                       ' & '.join(cell(h) for h in hdr) + r' \\', r'\midrule']
                for r in rows:
                    env.append(' & '.join(cell(c) for c in r) + r' \\')
                env += [r'\bottomrule', r'\end{tabular}', '}']
            out += env + ['']
            i = j; continue
        out.append(esc_text(inline(ln)))
        i += 1
    tex = '\n'.join(out)
    tex = re.sub(r'\n{3,}', '\n\n', tex)
    return tex

def main():
    os.makedirs(DST, exist_ok=True)
    for fn in sorted(os.listdir(SRC)):
        if not fn.endswith('.md'):
            continue
        with open(os.path.join(SRC, fn)) as f:
            md = f.read()
        is_appendix = fn.startswith(('09_', '10_'))
        if fn.startswith('00_'):
            # abstract: body only, no \section
            body = re.sub(r'^# Abstract\s*', '', md)
            body = re.sub(r'\s*<!--.*?-->', '', body, flags=re.S)
            tex = esc_text(inline(body)).strip() + '\n'
        else:
            tex = convert(md, is_appendix)
        stem = fn[:-3]
        with open(os.path.join(DST, stem + '.tex'), 'w') as f:
            f.write('% GENERATED by md2tex.py from sections/' + fn + ' - edit the .md, not this file\n')
            if fn.startswith('09_'):
                f.write('\\renewcommand{\\thefigure}{A\\arabic{figure}}\n'
                        '\\renewcommand{\\theHfigure}{A\\arabic{figure}}\n'
                        '\\setcounter{figure}{0}\n')
            f.write(tex + '\n')
        print('wrote', stem + '.tex')

if __name__ == '__main__':
    main()
