#!/usr/bin/env bash
# RAE-DiT IAD — DEFAULT preset: no memory optimizations.
#
#   fp32 · EMA on GPU · no activation recomputation · batch 2 x accum 4
#
# This is the fastest configuration per step, but DiT-XL + decoder + EMA + Adam
# in fp32 needs roughly > 30 GB of VRAM. On a 24 GB card use
# run_dit_iad_efficient.sh instead.
#
# Everything else is identical to run_dit_iad.sh (which this forwards to); all
# of its environment knobs and examples still apply:
#
#   EXPT="A B C" CATS="carpet screw" SEEDS="0 1" bash run_dit_iad_default.sh
#   STAGE=eval EXPT="B" MODES="single,multi,ode" T_VALUES="0.1,0.3,0.5" \
#     bash run_dit_iad_default.sh
#   BATCH=4 GRAD_ACCUM=2 bash run_dit_iad_default.sh   # bigger GPU, same eff. batch
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)

# MEM=full -> --no-grad-checkpoint --ema-device cuda  (config defaults)
export MEM=${MEM:-full}

echo "=== preset: default (no memory optimizations; MEM=$MEM) ==="
exec bash "$HERE/run_dit_iad.sh" "$@"
