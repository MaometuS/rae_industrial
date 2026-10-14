#!/usr/bin/env bash
# Experiment 3: seeds + score fusion (DINOv2-B; pixel, latent_self, mix; 3 seeds; held-out normals).
# Writes to results_exp3/ only. Experiment 2 results in results_latent_draem/ are never touched.
#   bash run_exp3.sh
#   CATS="screw" SEEDS="0" STEPS=300 OUT=results_exp3_smoke bash run_exp3.sh     # quick smoke test
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MVTEC_ROOT=${MVTEC_ROOT:-/home/maometus/Documents/datasets/mvtec_anomaly_detection}
RAE_ROOT=${RAE_ROOT:-$HERE/../rae_fidelity/RAE}
DTD_ROOT=${DTD_ROOT:-$HERE/data/dtd}
OUT=${OUT:-$HERE/results_exp3}
ENV_NAME=${ENV_NAME:-rae}
CATS=${CATS:-"carpet grid hazelnut screw transistor"}
MODES=${MODES:-"pixel latent_self mix"}
SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-3000}
N_VAL=${N_VAL:-20}

case "$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")" in
  "$HERE/results_latent_draem"*) echo "Refusing to write into the Experiment 2 folder"; exit 1 ;;
esac

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
[ -d "$RAE_ROOT/models" ] || { echo "RAE weights not found under $RAE_ROOT/models"; exit 1; }
mkdir -p "$OUT"

# shellcheck disable=SC2086
python "$HERE/latent_draem.py" \
  --rae-root "$RAE_ROOT" \
  --configs configs/stage1/pretrained/DINOv2-B.yaml \
  --mvtec-root "$MVTEC_ROOT" --dtd-root "$DTD_ROOT" --out "$OUT" \
  --categories $CATS --modes $MODES --seeds $SEEDS \
  --n-val "$N_VAL" --save-maps \
  --steps "$STEPS" --batch-size 16 \
  2>&1 | tee -a "$OUT/train.log"

python "$HERE/fuse_scores.py" --out "$OUT" 2>&1 | tee -a "$OUT/fusion.log"
echo "Reports: $OUT/report.md (per seed, raw scores)  $OUT/report_fusion.md (fusion, vs kNN)"
