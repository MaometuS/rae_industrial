"""Make ``iad`` importable from the tests directory (run from anywhere)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # rae_dit_iad/tests
ROOT = HERE.parent                              # rae_dit_iad/
RAE_ROOT = ROOT.parent / "rae_fidelity" / "RAE"
LATENT_DRAEM_DIR = ROOT.parent / "latent_draem"

for p in (str(ROOT), str(LATENT_DRAEM_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Official RAE repo on sys.path (only if it is present; pure-math tests do
# not need it and stay runnable without the clone).
src = RAE_ROOT / "src"
if src.is_dir() and str(src) not in sys.path:
    sys.path.insert(0, str(src))
