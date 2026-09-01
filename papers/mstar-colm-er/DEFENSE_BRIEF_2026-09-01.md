# Rank-1 defense brief — constant-memory recall (2026-09-01)

For Sam to own at a whiteboard. Every number below is from `brief.md` rows
C1–C11, regenerated today from the md5-pinned archives on the mini. The
agent attacks; Sam answers. Holes get logged at the bottom.

## The absolute claim (lead with this, per the headline rule)

A two-block fast-weight model whose entire memory is a fixed 32,768-byte
state (2 layers × 64×64 fp32) performs single-pass episodic recall at
K=32 with accuracy ≥0.998 in every seed at every horizon tested, from 454
to 1,798 tokens (8× context), with the state size held constant. A
parameter-matched flat-vector recurrence (512-byte state) and a
parameter-matched transformer read chance at every horizon, uncapped and
at every KV-cache budget from 1× to 32×, and the transformer stays at
chance after a 4-point learning-rate search.

Numbers: contender acc_A 0.99951 / 1.00000 / 0.99902 (seeds); ablation
0.0339 mean; transformer 0.0283 mean; chance 0.03125; demonstration bar
0.09375. Params 14,049,408 / 14,048,384 / 14,440,448. All arms 20,000
steps, lr 3e-4, K=32, identical episode seeds per (task, seed).

## Held fixed / grew / falsifier (the three sentences)

- HELD FIXED: parameter count (±2.8%), data, schedule, steps, episode
  seeds, K=32, the readout (the model's own forward, exact answer
  position), and the contender's state at 32,768 bytes.
- GREW: context length 454 → 902 → 1,798 tokens with the state fixed.
  Contender ≥0.998 at all three; baselines at chance at all three.
- FALSIFIER: any matched transformer config that clears the 0.094 bar on
  task 1 (the baseline-strengthening sweep launched today is that test),
  or a linear probe on the raw state that reads recall (none does; rf@0.9
  = 0.0 on taps i–iii; only the pre-LM-head hidden decodes, contender
  only, rf@0.9 0.674).

## Controls Sam must be able to name unprompted

1. Localization: S₀-zeroing collapses acc_A to 0.034/0.001/0.0002;
   S₁-zeroing leaves 0.9995/0.9949/0.9990. Recall lives in block 1's
   fast-weight state. 12/12 recurrent cells hard-stop clean.
2. The degenerate-baseline clause FIRED: the uncapped transformer is
   below the bar, so NO memory multiplier is quotable. The sanctioned
   sentence is "baseline non-competitive at matched params and tokens."
   Never "N× better." (This is what the strengthening sweep may change.)
3. Task 2 (multi-hop) is NON-DECISION-GRADE: contender 3/9 seeds clear,
   ablation 0/9, pooled CI spans zero, batch-effect flag (var-ratio 6.14).
   It is not in the headline and Sam must say so if asked.
4. K48 stress (K/d = 0.75): all three arms at chance. State it; it bounds
   the claim to K=32 at d=64 and is consistent with the capacity paper's
   located frontier (x0 = 0.5455 at d=64).
5. Training curves: contender CE 7.77 → 1.38; flat 7.83 → 7.69;
   transformer 7.84 → 7.51. The baselines optimized; they did not learn
   the task.

## Hostile-reviewer questions (agent asks these; log the answers)

Q1. Your transformer is 2 layers, 256 wide. Any real transformer does
this task trivially with attention. Why is this a fair baseline?
Q2. The transformer was LR-searched but never capacity- or length-
searched. Isn't "at chance" just under-training? (Answer must include
FIX-5's dissociation: lr=1e-4 optimizes best, loss 6.55, reads BELOW
chance.)
Q3. Episode-restricted recall with a delta rule is a known result
(DeltaNet, Schlag et al.). What is new? (Answer: the matched-baseline
separation and the fixed-state horizon result, not the mechanism.)
Q4. "Chance" for the transformer — could it be a readout bug? (Answer:
the same audited re-metric function scored every arm; the instrument
reproduced the contender's 0.9995.)
Q5. Why K=32 only? Where is the K sweep? (K48 all-chance; the capacity
paper carries the frontier; the recall paper is one operating point by
pre-registration.)
Q6. 3 seeds. CI? (Paired Δ CI (0.958, 0.973) vs ablation, (0.969, 0.974)
vs transformer; both exclude the pre-registered 0.30 margin by 3×.)
Q7. State bytes are constant but compute per token is not. Are you
hiding a FLOP advantage? (State it: per-token cost is O(d²) for the
contender; the transformer at 1,798 tokens is O(T·d); disclose both.)
Q8. Did the KV-cap protocol give the transformer a sink token? (Yes,
sink + FIFO; the uncapped arm is also at chance, so the cap is not the
cause.)

## Holes logged (fill during the pass)

- [ ]
