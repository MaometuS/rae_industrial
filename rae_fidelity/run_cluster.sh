#!/usr/bin/env bash
# RAE fidelity pilot on MVTec AD. Run on a GPU node.
#   MVTEC_ROOT=/data/mvtec_ad bash run_cluster.sh
set -euo pipefail

MVTEC_ROOT=/home/maometus/Documents/datasets/mvtec_anomaly_detection
WORK=${WORK:-$PWD}
RAE_ROOT=${RAE_ROOT:-$WORK/RAE}
OUT=${OUT:-$WORK/results_rae_fidelity}
HERE=$(cd "$(dirname "$0")" && pwd)
ENV_NAME=${ENV_NAME:-rae}
# CUDA build of torch 2.8.0: cu126 (driver >= 560), cu128 (>= 570), cu129 (>= 575). Check `nvidia-smi`.
CUDA_TAG=${CUDA_TAG:-cu128}

# 0) Dedicated conda env as the RAE README prescribes (python 3.10, torch 2.8.0).
#    Never install into anaconda base: an old torch there + new transformers breaks torchvision imports.
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"
python -m pip install -q uv

# 1) Official RAE repo + released decoders/stats (~a few GB)
if [ ! -d "$RAE_ROOT" ]; then
  git clone https://github.com/bytetriper/RAE.git "$RAE_ROOT"
fi
if ! python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.8') else 1)" 2>/dev/null; then
  uv pip install torch==2.8.0 torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
fi
uv pip install -r "$RAE_ROOT/requirements.txt"
uv pip install huggingface_hub diffusers scikit-image scikit-learn pandas scipy matplotlib tabulate

# Preflight: the exact import chain that failed before, plus GPU visibility
python - <<'EOF'
import sys, torch, torchvision, transformers
print(f"python {sys.version.split()[0]} | torch {torch.__version__} | torchvision {torchvision.__version__} "
      f"| transformers {transformers.__version__} | cuda {torch.cuda.is_available()}")
from torchvision import transforms  # noqa
assert torch.__version__.startswith("2.8"), "requirements.txt replaced torch; pin it back to 2.8.0"
assert torch.cuda.is_available(), "torch sees no GPU: wrong CUDA_TAG for this driver, or not on a GPU node"
EOF
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
