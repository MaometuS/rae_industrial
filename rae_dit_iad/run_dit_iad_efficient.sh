#!/usr/bin/env bash
# RAE-DiT IAD — EFFICIENT preset: all memory optimizations enabled.
#
#   fp32 (unchanged math) · activation recomputation on every DiT block
#   (and the decoder when it is fine-tuned) · EMA kept on CPU ·
#   frozen encoder offloaded to CPU after latent caching ·
#   batch 2 x accum 4 (effective batch 8, unchanged optimization semantics)
#
# Fits a 24 GB card (expected peaks ~15 GB for A, ~16 GB B/D, ~19 GB C/E).
# Costs ~30 % slower steps (recompute) plus ~0.1 s/step of PCIe traffic (EMA).
# None of these switches change the objective or the numerical result.
#
# Everything else is identical to run_dit_iad.sh (which this forwards to); all
# of its environment knobs and examples still apply:
#
#   EXPT="A B C" CATS="carpet screw" SEEDS="0 1" bash run_dit_iad_efficient.sh
#   STAGE=eval EXPT="B" MODES="single,multi,ode" T_VALUES="0.1,0.3,0.5" \
#     bash run_dit_iad_efficient.sh
#   MEM=full bash run_dit_iad_efficient.sh   # escape hatch: run this wrapper
#                                            # on a large GPU without recompute
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)

# MEM=low -> --grad-checkpoint --ema-device cpu
export MEM=${MEM:-low}
# limit allocator fragmentation (helps most in the tight 24 GB case)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "=== preset: efficient (activation recompute + CPU EMA; MEM=$MEM) ==="
exec bash "$HERE/run_dit_iad.sh" "$@"
