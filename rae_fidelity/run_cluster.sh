#!/usr/bin/env bash
# RAE fidelity pilot on MVTec AD. Run on a GPU node.
#   MVTEC_ROOT=/data/mvtec_ad bash run_cluster.sh
set -euo pipefail

MVTEC_ROOT=${MVTEC_ROOT:?set MVTEC_ROOT to the MVTec AD folder (contains bottle/, cable/, ...)}
WORK=${WORK:-$PWD}
RAE_ROOT=${RAE_ROOT:-$WORK/RAE}
OUT=${OUT:-$WORK/results_rae_fidelity}
HERE=$(cd "$(dirname "$0")" && pwd)

# 1) Official RAE repo + released decoders/stats (~a few GB)
if [ ! -d "$RAE_ROOT" ]; then
  git clone https://github.com/bytetriper/RAE.git "$RAE_ROOT"
fi
pip install -q -r "$RAE_ROOT/requirements.txt"
pip install -q huggingface_hub diffusers scikit-image scikit-learn pandas scipy matplotlib tabulate
if [ ! -d "$RAE_ROOT/models" ]; then
  (cd "$RAE_ROOT" && hf download nyu-visionx/RAE-collections --local-dir models)
fi

# 2) All released stage-1 (decoder) configs: DINOv2-B, SigLIP-B, MAE-B
mapfile -t CONFIGS < <(cd "$RAE_ROOT" && ls configs/stage1/pretrained/*.yaml)
echo "RAE configs: ${CONFIGS[*]}"

# Sanity check that one config loads and reconstructs (official script)
(cd "$RAE_ROOT" && python src/stage1_sample.py --config "${CONFIGS[0]}" \
   --image assets/pixabay_cat.png --output "$OUT/_sanity_recon.png") || {
  echo "Official stage1_sample.py failed - fix the RAE install before running the pilot"; exit 1; }

# 3) Pilot: all RAEs + SD-VAE baseline, 256px
python "$HERE/rae_fidelity.py" \
  --rae-root "$RAE_ROOT" \
  --configs "${CONFIGS[@]}" \
  --sdvae \
  --mvtec-root "$MVTEC_ROOT" \
  --out "$OUT" \
  --size 256 --batch-size 32 --max-train-good 50

echo "Report: $OUT/report.md   Grids: $OUT/<model>/grids/"
