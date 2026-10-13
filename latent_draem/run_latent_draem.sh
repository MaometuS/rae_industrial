#!/usr/bin/env bash
# Latent DRAEM pilot: DINOv2-B and MAE RAE latents, 5 MVTec categories, all synthesis modes.
#   bash run_latent_draem.sh                 (uses the `rae` env made by run_cluster.sh)
#   CATS="screw" MODES="pixel latent_tex" STEPS=500 bash run_latent_draem.sh    # quick smoke test
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MVTEC_ROOT=${MVTEC_ROOT:-/home/maometus/Documents/datasets/mvtec_anomaly_detection}
RAE_ROOT=${RAE_ROOT:-$HERE/../rae_fidelity/RAE}        # repo + models/ from the fidelity pilot
DTD_ROOT=${DTD_ROOT:-$HERE/data/dtd}                   # auto-downloaded (~600 MB) if empty
OUT=${OUT:-$HERE/results_latent_draem}
ENV_NAME=${ENV_NAME:-rae}
CATS=${CATS:-"carpet grid hazelnut screw transistor"}
MODES=${MODES:-"pixel latent_tex latent_self latent_noise mix"}
STEPS=${STEPS:-3000}

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
[ -d "$RAE_ROOT/models" ] || { echo "RAE weights not found under $RAE_ROOT/models (run run_cluster.sh first or set RAE_ROOT)"; exit 1; }

# shellcheck disable=SC2086
python "$HERE/latent_draem.py" \
  --rae-root "$RAE_ROOT" \
  --configs configs/stage1/pretrained/DINOv2-B.yaml configs/stage1/pretrained/MAE.yaml \
  --mvtec-root "$MVTEC_ROOT" \
  --dtd-root "$DTD_ROOT" \
  --out "$OUT" \
  --categories $CATS \
  --modes $MODES \
  --steps "$STEPS" --batch-size 16 \
  2>&1 | tee -a "$OUT.log"
