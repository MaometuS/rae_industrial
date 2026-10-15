#!/usr/bin/env bash
# Decoder-level ablation: RAE as a reconstruction-based detector, with masking (M), noisy bottleneck (N),
# synthetic anomalies (S) and all 8 combinations. Writes only to results_recon_ablation/.
#   bash run_recon_ablation.sh
#   CATS="screw" COMBOS="none MNS" STEPS=200 OUT=results_recon_smoke bash run_recon_ablation.sh   # smoke test
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MVTEC_ROOT=${MVTEC_ROOT:-/home/maometus/Documents/datasets/mvtec_anomaly_detection}
RAE_ROOT=${RAE_ROOT:-$HERE/../rae_fidelity/RAE}
DTD_ROOT=${DTD_ROOT:-$HERE/../latent_draem/data/dtd}
OUT=${OUT:-$HERE/results_recon_ablation}
ENV_NAME=${ENV_NAME:-rae}
CATS=${CATS:-"carpet grid hazelnut screw transistor"}
COMBOS=${COMBOS:-"none M N S MN MS NS MNS"}
SEEDS=${SEEDS:-"0"}
STEPS=${STEPS:-2000}

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
[ -d "$RAE_ROOT/models" ] || { echo "RAE weights not found under $RAE_ROOT/models"; exit 1; }
python "$HERE/../latent_draem/latent_draem.py" --help | grep -q -- "--n-val" || {
  echo "latent_draem/latent_draem.py is an old version (shared utilities); copy the current one first"; exit 1; }
mkdir -p "$OUT"

# shellcheck disable=SC2086
python "$HERE/rae_recon_ablation.py" \
  --rae-root "$RAE_ROOT" --config configs/stage1/pretrained/DINOv2-B.yaml \
  --mvtec-root "$MVTEC_ROOT" --dtd-root "$DTD_ROOT" --out "$OUT" \
  --categories $CATS --combos $COMBOS --seeds $SEEDS \
  --steps "$STEPS" --batch-size 8 --n-val 20 \
  2>&1 | tee -a "$OUT/train.log"
echo "Report: $OUT/report.md   Grids: $OUT/viz/"
