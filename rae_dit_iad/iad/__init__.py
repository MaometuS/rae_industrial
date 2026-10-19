"""RAE + original DiT (DiT-DH) for industrial anomaly detection.

Importing this package wires up the two shared project resources (paths only;
no code is modified):

* ``../latent_draem/latent_draem.py`` — MVTec split/loading + benchmark
  metrics (AUROC / AP / AUPRO), imported as ``latent_draem``.
* the official RAE repo — added to ``sys.path`` lazily via
  :func:`iad.models.setup_rae_root` when a config is loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

_LATENT_DRAEM_DIR = Path(__file__).resolve().parent.parent.parent / "latent_draem"
if str(_LATENT_DRAEM_DIR) not in sys.path:
    sys.path.insert(0, str(_LATENT_DRAEM_DIR))
