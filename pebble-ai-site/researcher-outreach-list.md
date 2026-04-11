# Researcher Outreach List — Matrix Thinking / Structured Representations
**Purpose:** After validating results with GPU credits, approach these researchers for expert feedback, collaboration, or mentorship.
**Strategy:** Lead with validated results + honest negative findings. Frame as complementary to their work.

---

## TIER 1: Most Directly Relevant (Bay Area Priority)

### Stanford — Quiet-STaR Team
- **Eric Zelikman** — Now at xAI (was Stanford). Co-author Quiet-STaR, STaR. Internal reasoning tokens.
  - [zelikman.me](https://zelikman.me/) | [X: @ezelikman](https://x.com/ezelikman)
  - **Why:** Quiet-STaR generates internal thought tokens before output — directly parallel to our matrix thought generation. Our matrix rank adds a measurable complexity axis his work lacks.

- **Noah D. Goodman** — Stanford (Psychology, Linguistics, CS). Co-author Quiet-STaR.
  - [CoCoLab](https://cocolab.stanford.edu/ndg)
  - **Why:** Computational models of reasoning. Could provide theoretical grounding for why matrix structure helps reasoning.

### Meta FAIR — COCONUT/CoCoMix Team
- **Sainbayar Sukhbaatar** — Research Scientist, FAIR. Co-author COCONUT.
  - [tesatory.github.io](https://tesatory.github.io/)
  - **Why:** COCONUT trains continuous thoughts. Our matrix thoughts are a structured version of this — rank gives measurable complexity that continuous vectors don't have.

- **Yuandong Tian** — Research Scientist Director, FAIR. Co-author COCONUT.
  - [X: @tydsh](https://x.com/tydsh)
  - **Why:** Leads reasoning/planning research at FAIR. Matrix thinking is a natural extension of COCONUT.

### Meta AI — BLT (Byte Latent Transformer) Team
- **Artidoro Pagnoni** — Meta AI. Lead author BLT.
  - [GitHub: artidoro](https://github.com/artidoro)
  - **Why:** BLT does byte-level processing with dynamic patching. Our matrix representations could replace their flat patches — combining byte-level with structured thought.

### UC San Diego — COCONUT Theory
- **Zhiting Hu** — Assistant Professor, UCSD. Co-author COCONUT.
  - [zhiting.ucsd.edu](https://zhiting.ucsd.edu/) | [X: @ZhitingHu](https://x.com/ZhitingHu)
  - **Why:** Works on theoretical foundations of latent reasoning. Could help formalize why rank enrichment matters.

- **Shibo Hao** — PhD Student (advisor: Zhiting Hu). First author COCONUT.
  - [ber666.github.io](https://ber666.github.io/)
  - **Why:** Did the actual implementation. Could give detailed technical feedback on our approach.

---

## TIER 2: Closely Related Work

### Google DeepMind — Perceiver IO
- **Andrew Jaegle** — DeepMind. Lead author Perceiver IO.
  - **Why:** Perceiver IO uses iterative cross-attention over structured inputs. Our matrix attention is a related approach to handling multi-domain data.

### Mila / U Montreal — Seq-VCR (Representation Collapse Prevention)
- **Md Rifat Arefin** — Mila. Lead author Seq-VCR.
  - [rarefin.github.io](https://rarefin.github.io/)
  - **Why:** Seq-VCR prevents the exact representation collapse we observed. Their variance-covariance regularization is in our planned training pipeline.

- **Irina Rish** — Professor, U Montreal. CIFAR Chair. Co-author Seq-VCR.
  - [irina-rish.com](https://www.irina-rish.com/)
  - **Why:** Senior researcher in reasoning + interpretability. High-profile endorsement would matter.

- **Ravid Shwartz-Ziv** — Meta / Hebrew U / ServiceNow. Co-author Seq-VCR.
  - [ravid-shwartz-ziv.com](https://www.ravid-shwartz-ziv.com/)
  - **Why:** Information theory in NNs. Our rank-as-complexity finding is an information-theoretic result.

### U Toronto / Vector Institute — LoopFormer
- **Ahmadreza Jeddi** — U Toronto. Lead author LoopFormer (ICLR 2026).
  - [GitHub: armenjeddi](https://github.com/armenjeddi)
  - **Why:** We directly compared against LoopFormer. Our matrix representation beats LoopFormer at T=1 by 175×. He'd want to see this.

- **Marco Ciccone** — Vector Institute. Co-author LoopFormer.
  - [marcociccone.github.io](https://marcociccone.github.io/)

### HKU NLP — EvaByte
- **Lingpeng Kong** — HKU NLP Group. EvaByte team.
  - [hkunlp.github.io](https://hkunlp.github.io/)
  - **Why:** EvaByte is the first competitive tokenizer-free byte LM. Our byte-level matrix model is a complementary approach.

### IBM Research — MBLM
- **Eric Egli, Matteo Manica, Jannis Born** — IBM Research Europe.
  - **Why:** MBLM uses hierarchical byte processing. Our matrix representations could replace their hierarchy with structured thoughts.

### Anthropic — Interpretability
- **Chris Olah** — Anthropic. Pioneer in neural network interpretability.
  - **Why:** Our rank enrichment finding is an interpretable signal of what happens during reasoning. His team could help formalize the mechanistic story.

---

## TIER 3: Broader Network

### Stanford
- **Christopher Manning** — Stanford NLP. Foundational NLP researcher.
  - [nlp.stanford.edu/~manning/](https://nlp.stanford.edu/~manning/)

### UC Berkeley
- **Dawn Song** — Professor, CS. AI safety + deep learning.
  - [people.eecs.berkeley.edu/~dawnsong/](https://people.eecs.berkeley.edu/~dawnsong/)

### Industry
- **Together AI Research Team** — Post-training, long-horizon reasoning.
  - [together.ai/research](https://www.together.ai/research)

---

## OUTREACH STRATEGY

1. **Lead with results, not asks.** Share the 22-experiment writeup + honest negative results. Researchers respect rigor over hype.
2. **Frame as extending their work.** "We took your COCONUT continuous thought idea and gave it a measurable complexity axis via matrix rank."
3. **Target grad students first.** They're more responsive and can broker introductions to advisors.
4. **Use Twitter/X.** Many are active. Sharing a clean results thread can get attention organically.
5. **Conference networking.** ICLR, NeurIPS, ICML in person. Bay Area workshops.
6. **Open-source the code.** Nothing builds credibility faster than reproducible results.

---

## TIMELINE
- **Now:** Get GPU credits, run core experiments
- **After Phase 2 results:** Reach out to Zelikman, Sukhbaatar, Hao (most directly relevant)
- **After Phase 3 cross-domain results:** Approach Pagnoni (BLT), Kong (EvaByte), IBM team
- **With publication-ready results:** Target senior researchers (Goodman, Rish, Manning, Olah)
