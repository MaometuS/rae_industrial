#!/usr/bin/env bash
# RAE-DiT industrial anomaly detection: full experiment matrix (A–E + noise ablations).
# Writes only to results_dit_iad/<exp_id>/.
#
# Usually launched through one of the two presets:
#   bash run_dit_iad_default.sh      # no memory optimizations (> 30 GB VRAM)
#   bash run_dit_iad_efficient.sh    # all memory optimizations (24 GB VRAM)
# This script is the shared engine; MEM picks the memory preset (auto/full/low).
#
#   bash run_dit_iad.sh                                                   # train+eval A–E, pilot cats, seed 0
#   EXPT="A B C" CATS="carpet screw" SEEDS="0 1" bash run_dit_iad.sh       # subset / more seeds
#   STAGE=eval EXPT="B" T_VALUES="0.1,0.3,0.5,0.7,0.9" bash run_dit_iad.sh # eval-only noise sweep
#   STAGE=train STEPS=500 EXPT="A" bash run_dit_iad.sh                     # quick smoke run
#   BATCH=1 GRAD_ACCUM=8 EXPT="A" bash run_dit_iad.sh                     # tighter memory
#   MEM=low bash run_dit_iad.sh                                           # force 24 GB preset
#   MEM=full BATCH=4 GRAD_ACCUM=2 bash run_dit_iad.sh                     # big GPU: fastest settings
#   (MEM defaults to auto: low below 30 GB VRAM, full above)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MVTEC_ROOT=${MVTEC_ROOT:-/home/maometus/Documents/datasets/mvtec_anomaly_detection}
RAE_ROOT=${RAE_ROOT:-$HERE/../rae_fidelity/RAE}
OUT=${OUT:-$HERE/results_dit_iad}
ENV_NAME=${ENV_NAME:-rae}
CATS=${CATS:-"carpet grid hazelnut screw transistor"}
SEEDS=${SEEDS:-"0"}
EXPT=${EXPT:-"A B C D E"}
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
T_VALUES=${T_VALUES:-"0.5"}       # noise level(s), comma list = sweep
REPEATS=${REPEATS:-1}             # repetitions for mode=multi
ODE_STEPS=${ODE_STEPS:-50}
VIZ=${VIZ:-6}
LAMBDA_FM=${LAMBDA_FM:-""}        # override hybrid weight (experiments D/E)

# letter -> config file
config_for() {
  case "$1" in
    A) echo "$HERE/configs/A_flow_matching.yaml" ;;
    B) echo "$HERE/configs/B_image_l1.yaml" ;;
    C) echo "$HERE/configs/C_image_l1_decfinetune.yaml" ;;
    D) echo "$HERE/configs/D_hybrid.yaml" ;;
    E) echo "$HERE/configs/E_hybrid_decfinetune.yaml" ;;
    *) echo "$HERE/configs/$1.yaml" ;;   # full basename, e.g. noise_fixed_t03
  esac
}

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
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
      [ -n "$LAMBDA_FM" ] && EXTRA="$EXTRA --lambda-fm $LAMBDA_FM"
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
      echo "=== eval exp=$exp seed=$SEED (mode=$MODES t=$T_VALUES) ==="
      # shellcheck disable=SC2086
      python "$HERE/rae_dit_iad.py" eval \
        --config "$CFG" --rae-root "$RAE_ROOT" --mvtec-root "$MVTEC_ROOT" \
        --out "$RUN_DIR" --categories $CATS --seed "$SEED" \
        --mode "$MODES" --t "$T_VALUES" --repeats "$REPEATS" \
        --ode-steps "$ODE_STEPS" --batch-size "$EVAL_BATCH" --viz "$VIZ" \
        2>&1 | tee -a "$RUN_DIR/eval_${exp}.log"
    fi

  done
done

echo "Reports: $OUT/seed*/**/report.md   CSVs: $OUT/seed*/**/results.csv   Grids: $OUT/seed*/**/viz/"
