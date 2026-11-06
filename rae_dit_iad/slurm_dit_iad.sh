#!/bin/bash
#SBATCH --job-name=RAE_DIT_IAD
###########RESOURCES###########
#SBATCH --partition=48-4
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
###############################
#SBATCH --output=RAE_DIT_IAD_%j.out
#SBATCH --error=RAE_DIT_IAD_%j.err
#SBATCH -v
#
# RAE-DiT IAD matrix on one 4-GPU node: 9 arms (A-G, J, K) + 3 noise ablations = 12 runs.
# The pipeline is single-process (one model per GPU, no DDP), so instead of torchrun
# each GPU gets its own queue of experiments and works through it with run_dit_iad.sh
# (train + eval, both pixel and latent_cos scores). Submit from rae_dit_iad/:
#
#   sbatch slurm_dit_iad.sh
#   SEEDS="0 1 2" sbatch slurm_dit_iad.sh             # three seeds (~3x the time)
#   SKIP_TESTS=1 STEPS=200 sbatch slurm_dit_iad.sh    # quick smoke run of everything
#   bash slurm_dit_iad.sh --setup-env                  # only build the conda env (login node)
#
# Everything the run needs is fetched automatically on first use (needs internet; if
# compute nodes are offline, run the --setup-env line once on the login node first):
#   - the official RAE code (bytetriper/RAE, pinned commit) into ../rae_fidelity/RAE
#     (the rae_industrial repo only stores a pointer to it, so a fresh clone is empty)
#   - the conda env (same recipe as ../rae_fidelity/run_cluster.sh)
#   - the three RAE checkpoints this pipeline loads + the DINOv2-B encoder (HF cache)

set -uo pipefail

# ---- paths (edit for your cluster) ------------------------------------------
ENV_NAME=${ENV_NAME:-rae}                       # conda env with torch 2.8 (README)
CONDA_SH=${CONDA_SH:-~/anaconda3/etc/profile.d/conda.sh}
export MVTEC_ROOT=${MVTEC_ROOT:-/po1/rakhimov/datasets/mvtec_anomaly_detection}
HERE=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
export RAE_ROOT=${RAE_ROOT:-$HERE/../rae_fidelity/RAE}
export OUT=${OUT:-$HERE/results_dit_iad}
export ENV_NAME
export SEEDS=${SEEDS:-"0"}
export STEPS=${STEPS:-""}                        # empty = config default (4000)

# ---- one queue per GPU (balanced: decoder fine-tuning C/E/G is the heaviest) --
QUEUES=(
  "A B noise_sigma005"
  "C D noise_sigma02"
  "E F noise_mask_all"
  "G J K"
)

# ---- conda env: use it if complete, otherwise create / complete it -----------
CUDA_TAG=${CUDA_TAG:-pypi}   # torch 2.8.0 source: pypi (bundles CUDA 12.8) | cu126 | cu128 | cu129
REQUIRED_IMPORTS="torch torchvision timm einops transformers omegaconf torchdiffeq sklearn scipy pandas yaml PIL pytest"

env_exists() { conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; }

env_missing_imports() {   # prints what is missing from the active env; empty = complete
  python - $REQUIRED_IMPORTS <<'PYCHECK'
import importlib, sys
missing = []
for m in sys.argv[1:]:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if "torch" not in missing:
    import torch
    if not torch.__version__.startswith("2.8"):
        missing.append(f"torch==2.8(found:{torch.__version__})")
print(" ".join(missing))
PYCHECK
}

install_env() {   # rae_fidelity/run_cluster.sh recipe: dedicated env, python 3.10, torch 2.8.0
  if [ "$ENV_NAME" = "base" ]; then
    echo "refusing to install into conda base (an old torch there breaks torchvision)"; return 1
  fi
  for h in pypi.org files.pythonhosted.org; do
    if ! getent hosts "$h" >/dev/null; then
      echo "no internet on $(hostname) (cannot resolve $h). Build the env once on the login node:"
      echo "  cd $HERE && bash slurm_dit_iad.sh --setup-env"
      return 1
    fi
  done
  [ -f "$RAE_ROOT/requirements.txt" ] || {
    echo "missing $RAE_ROOT/requirements.txt (is the RAE repo cloned there? set RAE_ROOT)"; return 1; }
  env_exists || conda create -n "$ENV_NAME" python=3.10 -y || return 1
  conda activate "$ENV_NAME" || return 1
  python -m pip install -q uv || return 1
  if ! python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.8') else 1)" 2>/dev/null; then
    if [ "$CUDA_TAG" != "pypi" ] && getent hosts download.pytorch.org >/dev/null; then
      uv pip install torch==2.8.0 torchvision==0.23.0 \
        --index-url "https://download.pytorch.org/whl/$CUDA_TAG" || return 1
    else
      uv pip install torch==2.8.0 torchvision==0.23.0 || return 1   # PyPI wheel bundles CUDA 12.8
    fi
  fi
  uv pip install -r "$RAE_ROOT/requirements.txt" || return 1
  uv pip install huggingface_hub diffusers scikit-image scikit-learn pandas scipy matplotlib \
    tabulate pytest pyyaml || return 1
}

ensure_env() {
  if env_exists; then
    conda activate "$ENV_NAME" || return 1
    MISSING=$(env_missing_imports)
    if [ -z "$MISSING" ]; then echo "conda env '$ENV_NAME' ready"; return 0; fi
    if [ "$ENV_NAME" != "rae" ] && [ -z "${ALLOW_ENV_INSTALL:-}" ]; then
      echo "env '$ENV_NAME' is missing: $MISSING"
      echo "not installing into a non-project env automatically (ALLOW_ENV_INSTALL=1 to allow)"
      return 1
    fi
    echo "env '$ENV_NAME' is missing: $MISSING -> installing"
  else
    echo "conda env '$ENV_NAME' not found -> creating it (python 3.10, torch 2.8.0; ~5-15 min)"
  fi
  # one setup at a time, even if several jobs start together
  local rc
  if command -v flock >/dev/null; then
    exec 9>"$HOME/.rae_env_setup_${ENV_NAME}.lock"; flock 9
    install_env; rc=$?
    flock -u 9; exec 9>&-
  else
    install_env; rc=$?
  fi
  [ "$rc" -eq 0 ] || return 1
  conda activate "$ENV_NAME" || return 1
  MISSING=$(env_missing_imports)
  if [ -n "$MISSING" ]; then echo "env '$ENV_NAME' still missing: $MISSING"; return 1; fi
  echo "conda env '$ENV_NAME' ready"
}

# ---- official RAE code + weights ---------------------------------------------
RAE_COMMIT=${RAE_COMMIT:-a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041}   # the commit the project used
RAE_FILES=(   # what the configs load (rae_config + rae_config_overrides)
  "decoders/dinov2/wReg_base/ViTXL_n08/model.pt"
  "stats/dinov2/wReg_base/imagenet1k/stat.pt"
  "DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt"
)

with_lock() {   # one setup at a time, even if several jobs start together
  if command -v flock >/dev/null; then
    ( flock 9; "$@" ) 9>"$HOME/.rae_industrial_setup.lock"
  else
    "$@"
  fi
}

ensure_rae() {
  [ -f "$RAE_ROOT/requirements.txt" ] && return 0
  if [ -d "$RAE_ROOT" ] && [ -n "$(ls -A "$RAE_ROOT" 2>/dev/null)" ]; then
    echo "$RAE_ROOT exists and is not empty, but has no requirements.txt; fix it or set RAE_ROOT"
    return 1
  fi
  getent hosts github.com >/dev/null || {
    echo "no internet on $(hostname) to clone RAE; run on the login node: bash slurm_dit_iad.sh --setup-env"
    return 1; }
  echo "cloning the official RAE repo (bytetriper/RAE @ ${RAE_COMMIT:0:7}) into $RAE_ROOT"
  rmdir "$RAE_ROOT" 2>/dev/null   # the empty placeholder left by the gitlink
  git clone -q https://github.com/bytetriper/RAE.git "$RAE_ROOT" &&
    git -C "$RAE_ROOT" checkout -q "$RAE_COMMIT"
}

ensure_weights() {   # needs the env active (huggingface_hub)
  local missing=() f
  for f in "${RAE_FILES[@]}"; do [ -f "$RAE_ROOT/models/$f" ] || missing+=("$f"); done
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "downloading RAE checkpoints from nyu-visionx/RAE-collections (${#missing[@]} files, a few GB)"
  fi
  python - "$RAE_ROOT/models" "${missing[@]}" <<'PYDL'
import sys
from huggingface_hub import hf_hub_download, snapshot_download
dest, files = sys.argv[1], sys.argv[2:]
for f in files:
    print("  ", f, flush=True)
    hf_hub_download("nyu-visionx/RAE-collections", f, local_dir=dest)
enc = "facebook/dinov2-with-registers-base"   # loaded by the RAE encoder at run time
try:
    snapshot_download(enc, local_files_only=True)
except Exception:
    print("   caching encoder", enc, flush=True)
    snapshot_download(enc)
PYDL
  for f in "${RAE_FILES[@]}"; do
    [ -f "$RAE_ROOT/models/$f" ] || { echo "still missing: $RAE_ROOT/models/$f"; return 1; }
  done
  echo "RAE weights ready under $RAE_ROOT/models"
}

with_lock ensure_rae || { echo "RAE code setup failed; nothing was run"; exit 1; }
source "$CONDA_SH" || { echo "cannot source conda from $CONDA_SH (set CONDA_SH)"; exit 1; }
set +u   # conda's activation scripts can reference unset variables
ensure_env || { echo "environment setup failed; nothing was run"; exit 1; }
with_lock ensure_weights || { echo "RAE weights setup failed; nothing was run"; exit 1; }
set -u
if [ "${1:-}" = "--setup-env" ]; then
  echo "RAE code, weights and env '$ENV_NAME' are ready. Submit with:  sbatch slurm_dit_iad.sh"; exit 0
fi
cd "$HERE"
for c in carpet grid hazelnut screw transistor; do
  [ -d "$MVTEC_ROOT/$c/train/good" ] || {
    echo "MVTec category '$c' not found under $MVTEC_ROOT (set MVTEC_ROOT)"; exit 1; }
done
mkdir -p "$OUT/logs"
echo "job $SLURM_JOB_ID on $(hostname)  env=$ENV_NAME  out=$OUT  seeds=$SEEDS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# ---- unit tests first (CPU, ~1 min): don't burn GPU hours on a broken build ---
if [ -z "${SKIP_TESTS:-}" ]; then
  if python -c "import pytest" 2>/dev/null; then
    python -m pytest tests/test_iad.py -q -p no:cacheprovider || {
      echo "unit tests failed; aborting (SKIP_TESTS=1 to override)"; exit 1; }
  else
    echo "pytest not installed in $ENV_NAME; skipping unit tests"
  fi
fi

# SLURM gives the job specific GPUs (e.g. CUDA_VISIBLE_DEVICES=2,3,5,6):
# worker i must use the i-th of those, not physical GPU i.
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [ "${#GPUS[@]}" -lt "${#QUEUES[@]}" ]; then
  echo "need ${#QUEUES[@]} GPUs, got ${#GPUS[@]} (${CUDA_VISIBLE_DEVICES:-unset})"; exit 1
fi

PIDS=()
for i in "${!QUEUES[@]}"; do
  (
    export CUDA_VISIBLE_DEVICES="${GPUS[$i]}"
    for exp in ${QUEUES[$i]}; do   # one exp at a time, so one failure doesn't stop the queue
      echo "[gpu ${GPUS[$i]}] start $exp $(date +%T)"
      if EXPT="$exp" bash "$HERE/run_dit_iad.sh"; then
        echo "[gpu ${GPUS[$i]}] done  $exp $(date +%T)"
      else
        echo "[gpu ${GPUS[$i]}] FAILED $exp $(date +%T)"
      fi
    done
  ) > "$OUT/logs/gpu${i}_${SLURM_JOB_ID:-local}.log" 2>&1 &
  PIDS+=($!)
  echo "worker $i -> GPU ${GPUS[$i]}: ${QUEUES[$i]}  (log: $OUT/logs/gpu${i}_${SLURM_JOB_ID:-local}.log)"
done

wait "${PIDS[@]}"
echo "=== summary ==="
grep -h "done\|FAILED" "$OUT"/logs/gpu*_"${SLURM_JOB_ID:-local}".log
grep -qh "FAILED" "$OUT"/logs/gpu*_"${SLURM_JOB_ID:-local}".log && STATUS=1 || STATUS=0
echo "reports: $OUT/seed*/<exp_id>/report.md"
set +u; conda deactivate
exit $STATUS
