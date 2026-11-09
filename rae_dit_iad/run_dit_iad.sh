#!/usr/bin/env bash
# RAE-DiT industrial anomaly detection: full experiment matrix (A-G, J, K + noise ablations).
# Corruption: SimpleNet-style (default) for every arm except A (original flow path).
# Writes only to results_dit_iad/<exp_id>/.
#
# Usually launched through one of the two presets:
#   bash run_dit_iad_default.sh      # no memory optimizations (> 30 GB VRAM)
#   bash run_dit_iad_efficient.sh    # all memory optimizations (24 GB VRAM)
# This script is the shared engine; MEM picks the memory preset (auto/full/low).
#
#   bash run_dit_iad.sh                                                   # train+eval A–G, pilot cats, seed 0
#   EXPT="A B C" CATS="carpet screw" SEEDS="0 1" bash run_dit_iad.sh       # subset / more seeds
#   STAGE=eval EXPT="A" T_VALUES="0.1,0.3,0.5,0.7,0.9" bash run_dit_iad.sh # eval-only noise sweep (flow path: A)
#   STAGE=train STEPS=500 EXPT="A" bash run_dit_iad.sh                     # quick smoke run
#   ADAPTOR=mlp EXPT="F" bash run_dit_iad.sh                               # swap the adaptor family
#   SIGMA=0.05 EXPT="B J" bash run_dit_iad.sh                             # SimpleNet-style noise level
#   SIGMA="0.5 1.5" EXPT="J" bash run_dit_iad.sh                          # sigma ~ U(0.5, 1.5) per sample
#   STAGE=eval EXPT="A B" SCORES="latent_l2" bash run_dit_iad.sh          # extra scoring, no retraining
#   BATCH=1 GRAD_ACCUM=8 EXPT="A" bash run_dit_iad.sh                     # tighter memory
#   MEM=low bash run_dit_iad.sh                                           # force 24 GB preset
#   MEM=full BATCH=4 GRAD_ACCUM=2 bash run_dit_iad.sh                     # big GPU: fastest settings
#   (MEM defaults to auto: low below 30 GB VRAM, full above)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MVTEC_ROOT=${MVTEC_ROOT:-/po1/rakhimov/datasets/mvtec_anomaly_detection}
RAE_ROOT=${RAE_ROOT:-$HERE/../rae_fidelity/RAE}
OUT=${OUT:-$HERE/results_dit_iad}
ENV_NAME=${ENV_NAME:-rae}
CATS=${CATS:-"carpet grid hazelnut screw transistor"}
SEEDS=${SEEDS:-"0"}
EXPT=${EXPT:-"A B C D E F G J K"}
STAGE=${STAGE:-"train eval"}      # "train", "eval" or "train eval"
STEPS=${STEPS:-""}                # empty = config default (4000)
BATCH=${BATCH:-2}                 # micro-batch
GRAD_ACCUM=${GRAD_ACCUM:-4}       # effective batch = BATCH * GRAD_ACCUM
EVAL_BATCH=${EVAL_BATCH:-8}       # no-grad inference can use a larger batch
# MEM=auto|full|low. full = fastest (no recompute, EMA on GPU); low = fits a 24 GB
# card (activation recompute + CPU EMA). auto picks 'low' below 30 GB of VRAM.
MEM=${MEM:-auto}
# reduce allocator fragmentation on 24 GB cards
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
MODES=${MODES:-"single"}          # single | multi | ode (comma list = sweep)
T_USER=${T_VALUES+set}            # did the caller set T_VALUES explicitly?
T_VALUES=${T_VALUES:-"0.5"}       # A only (flow path); SimpleNet arms default to their trained sigma
REPEATS=${REPEATS:-1}             # repetitions for mode=multi
ODE_STEPS=${ODE_STEPS:-50}
VIZ=${VIZ:-6}
LAMBDA_LAT=${LAMBDA_LAT:-""}      # override latent-term weight (experiments D/E)
ADAPTOR=${ADAPTOR:-""}            # override adaptor type (experiments F/G)
SIGMA=${SIGMA:-""}                # override SimpleNet-style noise sigma (every arm but A)
SDPA=${SDPA:-""}                  # attention kernels for training: auto | math (NaN debugging)
SCORES=${SCORES:-"pixel latent_cos disc"}  # each arm scored with each: pixel | latent_cos | latent_l2 | disc

# letter -> config file
config_for() {
  case "$1" in
    A) echo "$HERE/configs/A_flow_matching.yaml" ;;
    B) echo "$HERE/configs/B_image_l1.yaml" ;;
    C) echo "$HERE/configs/C_image_l1_decfinetune.yaml" ;;
    D) echo "$HERE/configs/D_image_l1_latent.yaml" ;;
    E) echo "$HERE/configs/E_image_l1_latent_decfinetune.yaml" ;;
    F) echo "$HERE/configs/F_image_l1_adaptor.yaml" ;;
    G) echo "$HERE/configs/G_image_l1_adaptor_decfinetune.yaml" ;;
    J) echo "$HERE/configs/J_latent_simplenet_nodecoder.yaml" ;;
    K) echo "$HERE/configs/K_latent_simplenet_adaptor.yaml" ;;
    J2) echo "$HERE/configs/J2_latent_sigma05_15.yaml" ;;
    K2) echo "$HERE/configs/K2_latent_adaptor_sigma05_15.yaml" ;;
    J3) echo "$HERE/configs/J3_latent_swap.yaml" ;;
    J4) echo "$HERE/configs/J4_latent_dtd.yaml" ;;
    J5) echo "$HERE/configs/J5_latent_mix.yaml" ;;
    K5) echo "$HERE/configs/K5_latent_adaptor_mix.yaml" ;;
    *) echo "$HERE/configs/$1.yaml" ;;   # full basename, e.g. noise_sigma02
  esac
}

# conda's activation scripts read unset variables (e.g. PS1 in batch jobs), which
# `set -u` turns into a fatal error: relax it around conda only. Skip activation
# entirely when the env is already active (slurm_dit_iad.sh activates it).
if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
  set +u
  CONDA_BIN=${CONDA_EXE:-$(command -v conda || true)}
  [ -n "$CONDA_BIN" ] || { echo "conda not found (source conda.sh first)"; exit 1; }
  eval "$("$CONDA_BIN" shell.bash hook)"
  conda activate "$ENV_NAME" || { echo "cannot activate conda env '$ENV_NAME'"; exit 1; }
  set -u
fi
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
[ -d "$RAE_ROOT/models" ] || { echo "RAE weights not found under $RAE_ROOT/models"; exit 1; }
python "$HERE/../latent_draem/latent_draem.py" --help >/dev/null || {
  echo "latent_draem/latent_draem.py not runnable; shared utilities unavailable"; exit 1; }
if [ "$MEM" = "auto" ]; then
  if python -c "import torch,sys; sys.exit(0 if torch.cuda.get_device_properties(0).total_memory < 30*1024**3 else 1)"; then
    MEM=low
  else
    MEM=full
  fi
fi
case "$MEM" in full|low) ;; *) echo "MEM must be auto, full or low (got '$MEM')"; exit 1 ;; esac
echo "memory preset: MEM=$MEM (grad_checkpoint=$([ "$MEM" = low ] && echo on || echo off), ema_device=$([ "$MEM" = low ] && echo cpu || echo cuda))"
mkdir -p "$OUT"

for exp in $EXPT; do
  CFG=$(config_for "$exp")
  [ -f "$CFG" ] || { echo "unknown experiment '$exp' (no $CFG)"; exit 1; }
  for SEED in $SEEDS; do
    # one directory per seed: checkpoints/config of different seeds never collide
    RUN_DIR="$OUT/seed${SEED}"
    mkdir -p "$RUN_DIR"

    if [[ " $STAGE " == *" train "* ]]; then
      # shellcheck disable=SC2086
      EXTRA=""
      [ -n "$STEPS" ] && EXTRA="$EXTRA --steps $STEPS"
      [ -n "$LAMBDA_LAT" ] && EXTRA="$EXTRA --lambda-latent $LAMBDA_LAT"
      [ -n "$ADAPTOR" ] && EXTRA="$EXTRA --adaptor $ADAPTOR"
      [ -n "$SIGMA" ] && EXTRA="$EXTRA --noise-sigma $SIGMA"
      [ -n "$SDPA" ] && EXTRA="$EXTRA --sdpa $SDPA"
      case "$MEM" in
        full) EXTRA="$EXTRA --no-grad-checkpoint --ema-device cuda" ;;
        low)  EXTRA="$EXTRA --grad-checkpoint --ema-device cpu" ;;
      esac
      echo "=== train exp=$exp seed=$SEED ==="
      # shellcheck disable=SC2086
      python "$HERE/rae_dit_iad.py" train \
        --config "$CFG" --rae-root "$RAE_ROOT" --mvtec-root "$MVTEC_ROOT" \
        --out "$RUN_DIR" --categories $CATS --seed "$SEED" --batch-size "$BATCH" \
        --grad-accum "$GRAD_ACCUM" $EXTRA \
        2>&1 | tee -a "$RUN_DIR/train_${exp}.log"
    fi

    if [[ " $STAGE " == *" eval "* ]]; then
      # SimpleNet-style arms (all but A) add no test-time noise: unless T_VALUES was set
      # explicitly, let eval use the checkpoint's own training sigma instead of 0.5
      T_ARG="--t $T_VALUES"; T_SHOW="$T_VALUES"
      if [ -z "$T_USER" ] && grep -q "^  corruption: simplenet" "$CFG"; then
        T_ARG=""; T_SHOW="trained sigma"
      fi
      for SC in $SCORES; do
        echo "=== eval exp=$exp seed=$SEED (mode=$MODES t=$T_SHOW score=$SC) ==="
        # shellcheck disable=SC2086
        python "$HERE/rae_dit_iad.py" eval \
          --config "$CFG" --rae-root "$RAE_ROOT" --mvtec-root "$MVTEC_ROOT" \
          --out "$RUN_DIR" --categories $CATS --seed "$SEED" \
          --mode "$MODES" $T_ARG --score "$SC" --repeats "$REPEATS" \
          --ode-steps "$ODE_STEPS" --batch-size "$EVAL_BATCH" --viz "$VIZ" \
          2>&1 | tee -a "$RUN_DIR/eval_${exp}.log"
      done
    fi

  done
done

echo "Reports: $OUT/seed*/**/report.md   CSVs: $OUT/seed*/**/results.csv   Grids: $OUT/seed*/**/viz/"
