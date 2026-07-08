#!/bin/bash
# Thin shell wrapper (house convention, same as every *_chain.sh in this
# directory) around run_k69_s1733_contingency.py -- the registered K=69/
# d=96 contingency seed (1733) cell launch. TMUX-READY: the ORCHESTRATOR
# (the RUN agent) launches THIS inside a named tmux session
# (keyanchor_k69_contingency), not a backgrounded SSH shell -- a local
# Claude Code session restart cannot kill it.
#
# GATE 0a/0b (mirrors keyanchor_scaling_wide_chain.sh's own GATE 0a/0b
# exactly): both PI-decision signoff tokens must already be on record in
# the environment this script is invoked from -- this script does NOT set
# them itself (no silent default), matching keyanchor_scaling_wide_chain.sh
# and run_k69_s1733_contingency.py's own explicit (a1)/(a2) checks.
set -euo pipefail
cd /home/nvidia/chapter2/deltanet_rd
: "${KEYANCHOR_SCALING_PI_SIGNOFF:?KEYANCHOR_SCALING_PI_SIGNOFF=1 must be set in the environment}"
: "${KEYANCHOR_SCALING_EXT_PI_SIGNOFF:?KEYANCHOR_SCALING_EXT_PI_SIGNOFF=1 must be set in the environment}"
/home/nvidia/tdenv/bin/python run_k69_s1733_contingency.py
