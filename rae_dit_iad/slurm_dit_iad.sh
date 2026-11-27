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
# RAE-DiT IAD runs on one 4-GPU node. Default (PLAN=ad2): the best configuration (J5,
# J5 on a 32x32 grid) and the kNN control on MVTec AD 2. PLAN=locpx: the pixel-head arm of
# PLAN=loc again (job 163739 lost it). PLAN=loc: localization ablation on 5 MVTec
# categories (J5, J5 + pixel head, J5 on a 32x32 grid). PLAN=loco: MVTec LOCO (J5, A, J3, kNN). PLAN=all: every experiment, one
# DiT per category. PLAN=structured: J3-J5, K5 (+ A, J2). PLAN=next: J2, K2, A@1, A@2.
# PLAN=matrix: the 9 arms (A-G, J, K) + 3 noise ablations = 12 runs.
# The pipeline is single-process (one model per GPU, no DDP), so instead of torchrun
# each GPU gets its own queue of experiments and works through it with run_dit_iad.sh
# (train + eval, both pixel and latent_cos scores). Submit from rae_dit_iad/:
#
#   sbatch slurm_dit_iad.sh                           # PLAN=ad2: MVTec AD 2, J5 J5_grid32 kNN
#   PLAN=draem sbatch slurm_dit_iad.sh                # exact DRAEM on the 5 loc categories
#   PLAN=draemdec sbatch slurm_dit_iad.sh             # the same with the decoder fine-tuned
#   PLAN=locpx sbatch slurm_dit_iad.sh                # J5_pixel_head rerun
#   PLAN=loc sbatch slurm_dit_iad.sh                  # localization ablation
#   PLAN=loco sbatch slurm_dit_iad.sh                 # MVTec LOCO, J5 A J3 + kNN
#   PLAN=full sbatch slurm_dit_iad.sh                 # J5, K5 on all 15 MVTec categories
#   PLAN=seeds sbatch slurm_dit_iad.sh                # J5 and K5 at seeds 1, 2 (job 158993)
#   PLAN=samelr sbatch slurm_dit_iad.sh               # A@2 J J3 K5 at lr 2e-5 (job 158355)
#   PLAN=retry sbatch slurm_dit_iad.sh                # the 7 arms job 156032 lost (job 158158)
#   PLAN=all sbatch slurm_dit_iad.sh                  # 20 runs x 2 categories (carpet, screw)
#   CATS="carpet grid hazelnut screw transistor" sbatch slurm_dit_iad.sh  # the 5 pilot categories
#   STEPS=2000 sbatch slurm_dit_iad.sh                # half the steps per category model
#   SCOPE=joint PLAN=next sbatch slurm_dit_iad.sh     # the multi-class (joint) runs
#   PLAN=next sbatch slurm_dit_iad.sh                 # J2 K2 A@1 A@2
#   PLAN=matrix sbatch slurm_dit_iad.sh               # the full A-K matrix again
#   SEEDS="0 1 2" PLAN=matrix sbatch slurm_dit_iad.sh # three seeds (~3x the time)
#   SKIP_TESTS=1 STEPS=200 sbatch slurm_dit_iad.sh    # quick smoke run of everything
#   bash slurm_dit_iad.sh --setup-env                  # only build the conda env (login node)
#   STAGE=eval ONLY="A C" sbatch slurm_dit_iad.sh      # re-evaluate some arms, no training
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

# ---- one queue per GPU ----------------------------------------------------------
# An entry is an experiment (letter or config basename), optionally with @SEED to
# run that entry at one seed only (overrides SEEDS), e.g. "A@1", and/or with :CATS to
# run it on its own comma-separated categories (overrides CATS), e.g. "J5:tile,wood".
#   PLAN=ad2    (default) MVTec AD 2 (8 categories, AD2_ROOT; test_public only, the
#               private splits have no labels): J5 (one GPU, all 8), J5_grid32 (split over
#               three GPUs) and the frozen-token kNN control. Each checkpoint is scored with
#               AD2_SCORES at AD2_BLURS before it is deleted. Rows add AUPRO@0.05 and pixel
#               F1-max (MVTec AD 2's AU-PRO_0.05 / SegF1 counterparts). Results in
#               results_dit_iad_ad2/, checkpoints in CKPT_ROOT/../rae_dit_iad_ad2.
#   PLAN=draem  exact DRAEM with the RAE DiT as restorer (DR_draem; DR_draem_lat adds
#               the latent term) on the 5 localization categories, scored with DRAEM_SCORES
#               at LOC_BLURS; results join results_dit_iad_loc/ next to J5 and J5_grid32.
#   PLAN=draemdec  PLAN=draem with the RAE decoder fine-tuned (DR_draem_dec,
#               DR_draem_lat_dec; decoder lr = 0.1 x DiT lr). With PLAN=draem's results
#               this completes latent term on/off x decoder frozen/fine-tuned; same
#               categories, scores, blurs and results_dit_iad_loc/.
#   PLAN=locpx  J5_pixel_head of PLAN=loc again, split over 3 GPUs: job 163739
#               stopped it on a device mismatch in the head's grad-norm (fixed); J5 and
#               J5_grid32 finished there and keep their results (same results_dit_iad_loc/)
#   PLAN=loc    localization ablation on the 5 MVTec AD categories where J5
#               localizes worst (metal_nut, tile, transistor, wood, cable; pixel AUROC
#               0.86-0.90): J5 retrained (seed 0, its checkpoints were deleted), J5 +
#               pixel head, J5 on a 32x32 grid (split over two GPUs). Every checkpoint
#               is scored with LOC_SCORES at blur sigmas LOC_BLURS before it is deleted:
#               oracles, shift TTA, guided filter, decoder difference, fusions, pixel
#               head. Results in results_dit_iad_loc/.
#   PLAN=loco   MVTec LOCO AD (5 categories, LOCO_ROOT): J5 (split over two GPUs), A,
#               J3 and the frozen-token kNN control. Results in results_dit_iad_loco/,
#               checkpoints in CKPT_ROOT/../rae_dit_iad_loco: never mixed with MVTec AD.
#   PLAN=full   job 159148: J5 and K5 at seed 0 on the 13 MVTec categories not run yet
#               (carpet + screw are kept), each arm split over two GPUs; KEEP_RESULTS=1
#               so the workers add to the arm's existing results.csv
#   PLAN=seeds  job 158993: J5 and K5 at seeds 1 and 2 (carpet + screw, lr 2e-5): do they
#               beat A (0.933 +- 0.009 over 3 seeds) beyond seed noise?
#   PLAN=samelr job 158355: A@2, J, J3, K5: the latent arms job 156032 finished at lr 1e-4,
#               rerun at 2e-5 so every latent arm shares one lr (carpet + screw)
#   PLAN=retry  job 158158: only the 7 arms job 156032 stopped on the non-finite guard
#               (A seeds 0-1, J2, J4, J5, K, K2), at lr 2e-5. Its finished arms keep
#               their results in place (each arm writes only to its own <exp_id>/).
#   PLAN=all    every experiment so far, one DiT per category (SCOPE,
#               default per_category): A (seeds 0-2), B-G, J, K, the 3 noise
#               ablations, J2, K2, J3-J5, K5 = 20 runs. Most informative first in
#               each queue, so a job cut short still has the key arms.
#   PLAN=structured J3 J4 J5 K5 (+ A, J2)
#   PLAN=next   job 153649: J2, K2 (sigma ~ U(0.5, 1.5)) and A at seeds 1 and 2
#   PLAN=matrix the full A-K matrix + noise ablations of job 152776
# MVTec categories: one texture (carpet) + one object (screw) by default
export CATS=${CATS:-"carpet screw"}
PLAN=${PLAN:-ad2}
# localization ablation (PLAN=loc): every spec is one row per blur sigma, from shared
# DiT passes (iad/infer.py score_maps). Specs a checkpoint cannot compute (pxhead
# without a pixel head) are skipped with a message.
LOC_SCORES=${LOC_SCORES:-"latent_cos,disc,pixel,dec_diff,fuse_dec,fuse_pixel,fuse_disc,oracle16,oracle32,oracle16+gf,latent_cos+s4,latent_cos+s16,disc+s4,disc+s16,fuse_dec+s4,latent_cos+gf,disc+gf,fuse_dec+gf,disc+s4+gf,pxhead,pxhead+s4"}
LOC_BLURS=${LOC_BLURS:-"0,1,2,4,8"}
# MVTec AD 2 (PLAN=ad2): the winners of PLAN=loc (latent_cos for images; fuse_disc, disc,
# disc+gf for maps), dec_diff as their ingredient, and the oracles as the resolution ceiling
AD2_SCORES=${AD2_SCORES:-"latent_cos,disc,dec_diff,fuse_disc,disc+gf,fuse_disc+gf,oracle16,oracle32"}
AD2_BLURS=${AD2_BLURS:-"0,2,4,8"}
# exact DRAEM (PLAN=draem): the U-Net's map, and the restoration read the usual ways
DRAEM_SCORES=${DRAEM_SCORES:-"draem,draem+gf,latent_cos,pixel,dec_diff,oracle16"}
case "$PLAN" in
  draem)      QUEUES=("DR_draem:metal_nut,tile,transistor" "DR_draem:wood,cable"
                      "DR_draem_lat:metal_nut,tile,transistor" "DR_draem_lat:wood,cable")
              export OUT=${OUT_LOC:-$HERE/results_dit_iad_loc}
              CATS="metal_nut tile transistor wood cable"
              export SCORES=$DRAEM_SCORES BLUR_SWEEP=$LOC_BLURS
              export KEEP_RESULTS=1 ;;
  draemdec)   QUEUES=("DR_draem_dec:metal_nut,tile,transistor" "DR_draem_dec:wood,cable"
                      "DR_draem_lat_dec:metal_nut,tile,transistor" "DR_draem_lat_dec:wood,cable")
              export OUT=${OUT_LOC:-$HERE/results_dit_iad_loc}
              CATS="metal_nut tile transistor wood cable"
              export SCORES=$DRAEM_SCORES BLUR_SWEEP=$LOC_BLURS
              export KEEP_RESULTS=1 ;;
  ad2)        QUEUES=("J5" "J5_grid32:can,fabric,fruit_jelly" "J5_grid32:rice,sheet_metal,vial"
                      "J5_grid32:wallplugs,walnuts knn")
              export MVTEC_ROOT=${AD2_ROOT:-/po1/rakhimov/datasets/mvtec_ad_2/mvtec_ad_2}
              export OUT=${OUT_AD2:-$HERE/results_dit_iad_ad2}
              export CKPT_ROOT=${CKPT_ROOT:-/po1/rakhimov/training/rae_dit_iad_ad2}
              CATS="can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts"
              export SCORES=$AD2_SCORES BLUR_SWEEP=$AD2_BLURS
              export KEEP_RESULTS=1 ;;
  locpx)      QUEUES=("J5_pixel_head:metal_nut,tile" "J5_pixel_head:transistor,wood" "J5_pixel_head:cable")
              export OUT=${OUT_LOC:-$HERE/results_dit_iad_loc}
              CATS="metal_nut tile transistor wood cable"
              export SCORES=$LOC_SCORES BLUR_SWEEP=$LOC_BLURS
              export KEEP_RESULTS=1 ;;
  loc)        QUEUES=("J5" "J5_pixel_head" "J5_grid32:metal_nut,tile,transistor" "J5_grid32:wood,cable")
              export OUT=${OUT_LOC:-$HERE/results_dit_iad_loc}
              CATS="metal_nut tile transistor wood cable"
              export SCORES=$LOC_SCORES BLUR_SWEEP=$LOC_BLURS
              export KEEP_RESULTS=1 ;;
  loco)       QUEUES=("J5:breakfast_box,juice_bottle,pushpins" "J5:screw_bag,splicing_connectors" "A" "J3 knn")
              export MVTEC_ROOT=${LOCO_ROOT:-/po1/rakhimov/datasets/mvtec_loco}
              export OUT=${OUT_LOCO:-$HERE/results_dit_iad_loco}
              export CKPT_ROOT=${CKPT_ROOT:-/po1/rakhimov/training/rae_dit_iad_loco}
              CATS="breakfast_box juice_bottle pushpins screw_bag splicing_connectors"
              export KEEP_RESULTS=1 ;;
  full)       QUEUES=("J5:bottle,cable,capsule,grid,hazelnut,leather,metal_nut" "J5:pill,tile,toothbrush,transistor,wood,zipper" "K5:bottle,cable,capsule,grid,hazelnut,leather,metal_nut" "K5:pill,tile,toothbrush,transistor,wood,zipper")
              CATS="carpet screw bottle cable capsule grid hazelnut leather metal_nut pill tile toothbrush transistor wood zipper"
              export KEEP_RESULTS=1 ;;
  seeds)      QUEUES=("J5@1" "J5@2" "K5@1" "K5@2") ;;
  samelr)     QUEUES=("A@2" "J" "J3" "K5") ;;
  retry)      QUEUES=("A@0 J4" "A@1 J5" "K J2" "K2") ;;
  all)        QUEUES=("A J B C D"
                      "A@1 K E F G"
                      "A@2 J2 noise_sigma005 noise_sigma02 noise_mask_all"
                      "J3 J4 J5 K5 K2") ;;
  structured) QUEUES=("J3 A" "J4 J2" "J5" "K5") ;;
  next)       QUEUES=("J2" "K2" "A@1" "A@2") ;;
  matrix)     QUEUES=("A B noise_sigma005" "C D noise_sigma02" "E F noise_mask_all" "G J K") ;;
  *) echo "unknown PLAN '$PLAN' (ad2 | draem | draemdec | locpx | loc | loco | full | seeds | samelr | retry | all | structured | next | matrix)"; exit 1 ;;
esac

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

# ---- heavy files: checkpoints outside the repo, deleted after evaluation ---------
export CKPT_ROOT=${CKPT_ROOT:-/po1/rakhimov/training/rae_dit_iad}
export KEEP_CKPT=${KEEP_CKPT:-0}
mkdir -p "$CKPT_ROOT" && [ -w "$CKPT_ROOT" ] \
  || { echo "cannot write checkpoints to $CKPT_ROOT (set CKPT_ROOT)"; exit 1; }
echo "checkpoints: $CKPT_ROOT (KEEP_CKPT=$KEEP_CKPT: $([ "$KEEP_CKPT" = 0 ] \
  && echo 'deleted after each evaluation' || echo kept))"

# ---- DTD textures (noise.kind dtd / mix: J4, J5, K5) ------------------------------
# latent_draem's download folder by default (already there if latent_draem ran).
if [ -z "${DTD_ROOT:-}" ]; then
  DTD_ROOT=$(cd "$HERE/.." && pwd)/latent_draem/data/dtd
  # not there yet: download to the heavy-files area instead of the repo
  [ -n "$(find "$DTD_ROOT" -type f -name '*.jpg' -print -quit 2>/dev/null)" ] \
    || DTD_ROOT=/po1/rakhimov/training/dtd
fi
export DTD_ROOT
ensure_dtd() {   # needs the env active (torchvision); downloads ~600 MB once
  # run inside latent_draem/: from the repo root `latent_draem` is the folder (a
  # namespace package), not latent_draem.py, and the import fails
  mkdir -p "$DTD_ROOT"
  ( cd "$HERE/../latent_draem" && python - "$DTD_ROOT" <<'PYDTD'
import sys
from pathlib import Path
from latent_draem import get_dtd_images
root = Path(sys.argv[1])
try:
    n = len(get_dtd_images(root))
except Exception as exc:  # e.g. no network on the node: say so, not just "failed"
    print(f"DTD download/check under {root} failed: {type(exc).__name__}: {exc}")
    sys.exit(1)
print(f"DTD ready under {root}: {n} images")
sys.exit(0 if n else 1)
PYDTD
  )
}
if [[ " ${QUEUES[*]} " =~ [[:space:]](J4|J5|K5|J5_[a-z0-9_]+|DR_[a-z0-9_]+)(@[0-9]+)?(:[^[:space:]]*)?[[:space:]] ]]; then
  with_lock ensure_dtd || { echo "DTD setup failed (set DTD_ROOT); nothing was run"; exit 1; }
fi
set -u
if [ "${1:-}" = "--setup-env" ]; then
  echo "RAE code, weights and env '$ENV_NAME' are ready. Submit with:  sbatch slurm_dit_iad.sh"; exit 0
fi
cd "$HERE"
for c in $CATS; do
  [ -d "$MVTEC_ROOT/$c/train/good" ] || {
    echo "MVTec category '$c' not found under $MVTEC_ROOT (set MVTEC_ROOT)"; exit 1; }
done
if [ "$PLAN" = ad2 ]; then  # layout + masks check before any GPU time is spent
  python - "$MVTEC_ROOT" $CATS <<'PYAD2' || { echo "MVTec AD 2 layout check failed; nothing was run"; exit 1; }
import sys
from pathlib import Path
from iad import datasets as D
root = Path(sys.argv[1])
for cat in sys.argv[2:]:
    if not D.is_ad2(root, cat):
        sys.exit(f"{cat}: no {D.AD2_TEST}/ under {root}")
    tr, va, te = D.split(root, cat)
    bad = sum(d != "good" for _p, d, _m in te)
    print(f"AD2 {cat}: train {len(tr)}  val {len(va)}  test_public {len(te)} ({bad} bad)")
    if not tr or not bad or bad == len(te):
        sys.exit(f"{cat}: empty train split or test_public without both good and bad images")
PYAD2
fi
mkdir -p "$OUT/logs"
echo "job $SLURM_JOB_ID on $(hostname)  env=$ENV_NAME  out=$OUT  seeds=$SEEDS  cats=$CATS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# ---- unit tests first (CPU, ~1 min): don't burn GPU hours on a broken build ---
if [ -z "${SKIP_TESTS:-}" ]; then
  if python -c "import pytest" 2>/dev/null; then
    python -m pytest tests/test_iad.py tests/test_datasets.py -q -p no:cacheprovider || {
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
    for item in ${QUEUES[$i]}; do  # one exp at a time, so one failure doesn't stop the queue
      spec=${item%%:*}                  # EXP[@SEED]
      cats=$CATS
      [[ "$item" == *:* ]] && cats=${item#*:} && cats=${cats//,/ }
      exp=${spec%@*}
      seeds=$SEEDS
      [[ "$spec" == *@* ]] && seeds=${spec#*@}
      if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $exp "* && " $ONLY " != *" $item "* ]]; then
        continue
      fi
      echo "[gpu ${GPUS[$i]}] start $item $(date +%T)"
      if EXPT="$exp" SEEDS="$seeds" CATS="$cats" bash "$HERE/run_dit_iad.sh"; then
        echo "[gpu ${GPUS[$i]}] done  $item $(date +%T)"
      else
        echo "[gpu ${GPUS[$i]}] FAILED $item $(date +%T)"
      fi
    done
  ) > "$OUT/logs/gpu${i}_${SLURM_JOB_ID:-local}.log" 2>&1 &
  PIDS+=($!)
  echo "worker $i -> GPU ${GPUS[$i]}: ${QUEUES[$i]}  (log: $OUT/logs/gpu${i}_${SLURM_JOB_ID:-local}.log)"
done

wait "${PIDS[@]}"
echo "=== summary ==="
grep -h "done\|FAILED\|non-finite gradients:" "$OUT"/logs/gpu*_"${SLURM_JOB_ID:-local}".log
grep -qh "FAILED" "$OUT"/logs/gpu*_"${SLURM_JOB_ID:-local}".log && STATUS=1 || STATUS=0
if [ "$STATUS" -ne 0 ]; then
  echo "=== last lines before each failure ==="
  for f in "$OUT"/logs/gpu*_"${SLURM_JOB_ID:-local}".log; do
    grep -B 8 "FAILED" "$f" | grep -v "^\[gpu .*\] \(start\|done\)" | head -40
  done
fi
echo "reports: $OUT/${SCOPE:-per_category}/seed*/<exp_id>/report.md (joint: $OUT/seed*/)"
set +u; conda deactivate
exit $STATUS
