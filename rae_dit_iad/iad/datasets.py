"""Dataset layouts: MVTec AD and MVTec LOCO AD (torch-free).

The layout is detected per category from the folders, so every command takes the
dataset root through the same ``--mvtec-root`` flag.

MVTec AD::

    <cat>/train/good/*.png
    <cat>/test/<defect>/*.png
    <cat>/ground_truth/<defect>/<stem>_mask.png

MVTec LOCO AD::

    <cat>/train/good/*.png
    <cat>/validation/good/*.png           official normal validation split
    <cat>/test/{good,logical_anomalies,structural_anomalies}/*.png
    <cat>/ground_truth/<type>/<stem>/000.png, 001.png, ...   one file per defect
    <cat>/defects_config.json             per-defect sPRO saturation thresholds

A test entry is ``(path, defect_type, masks)`` where ``masks`` is ``None`` (normal),
one ``Path`` (MVTec AD) or a list of ``Path`` (LOCO, one per defect).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")  # = latent_draem: same files, same order

LOCO_CATEGORIES = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag",
                   "splicing_connectors"]
LOCO_TYPES = ("logical_anomalies", "structural_anomalies")
LOCO_SHORT = {"logical_anomalies": "logical", "structural_anomalies": "structural"}

MaskSpec = Union[None, Path, List[Path]]
TestEntry = Tuple[Path, str, MaskSpec]


def list_images(d: Path) -> List[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def is_loco(root: Path, cat: str) -> bool:
    c = Path(root) / cat
    return (c / "validation" / "good").is_dir() and any(
        (c / "test" / t).is_dir() for t in LOCO_TYPES)


def split(root: Path, cat: str) -> Tuple[List[Path], Optional[List[Path]], List[TestEntry]]:
    """``(train, val, test)``. ``val`` is the official normal validation split
    (LOCO), or ``None`` when the dataset has none (MVTec AD: carve it from train)."""
    root = Path(root)
    c = root / cat
    if not is_loco(root, cat):
        train = list_images(c / "train" / "good")
        test: List[TestEntry] = []
        tdir = c / "test"
        for d in sorted(p for p in tdir.iterdir() if p.is_dir()) if tdir.is_dir() else []:
            for p in list_images(d):
                m: MaskSpec = None
                if d.name != "good":
                    cand = c / "ground_truth" / d.name / f"{p.stem}_mask.png"
                    m = cand if cand.exists() else None
                test.append((p, d.name, m))
        return train, None, test

    train = list_images(c / "train" / "good")
    val = list_images(c / "validation" / "good")
    test = []
    for d in sorted(p for p in (c / "test").iterdir() if p.is_dir()):
        for p in list_images(d):
            m = None
            if d.name != "good":
                files = list_images(c / "ground_truth" / d.name / p.stem)
                m = files or None
            test.append((p, d.name, m))
    return train, val, test


def load_mask_any(m: MaskSpec, size: int) -> np.ndarray:
    """Binary mask at ``size x size`` (nearest); LOCO's per-defect files are OR-ed."""
    if m is None:
        return np.zeros((size, size), dtype=bool)
    files = m if isinstance(m, (list, tuple)) else [m]
    out = np.zeros((size, size), dtype=bool)
    thr = 0 if isinstance(m, (list, tuple)) else 127  # LOCO: any defect value; AD: binary
    for f in files:
        img = Image.open(f).convert("L").resize((size, size), Image.NEAREST)
        out |= np.asarray(img) > thr
    return out


# ---------------------------------------------------------------------------
# sPRO (MVTec LOCO): saturated per-region overlap
# ---------------------------------------------------------------------------

def loco_defects_config(root: Path, cat: str) -> Dict[int, Tuple[float, bool]]:
    """``{pixel_value: (saturation_threshold, relative)}`` from ``defects_config.json``.

    ``relative``: the threshold is a fraction of the defect's own area; otherwise
    it is an absolute pixel count at the native image resolution."""
    f = Path(root) / cat / "defects_config.json"
    if not f.is_file():
        return {}
    out: Dict[int, Tuple[float, bool]] = {}
    for d in json.loads(f.read_text()):
        out[int(d["pixel_value"])] = (float(d["saturation_threshold"]),
                                      bool(d.get("relative_saturation", False)))
    return out


def loco_defects(
    masks: Sequence[MaskSpec],
    size: int,
    config: Dict[int, Tuple[float, bool]],
) -> List[Tuple[int, np.ndarray, float]]:
    """``[(image_index, region, saturation_pixels)]``, one per LOCO defect file.

    Regions are resized to ``size x size`` (nearest). An absolute saturation
    threshold is rescaled by the area ratio ``size^2 / (H * W)`` of the native
    image; every threshold is capped at the region's own area, so a fully covered
    defect always counts as 1.
    """
    out: List[Tuple[int, np.ndarray, float]] = []
    for i, m in enumerate(masks):
        if not isinstance(m, (list, tuple)):
            continue
        for f in m:
            native = np.asarray(Image.open(f).convert("L"))
            vals = native[native > 0]
            if vals.size == 0:
                continue
            pv = int(np.bincount(vals).argmax())
            region = np.asarray(Image.fromarray(native).resize((size, size), Image.NEAREST)) > 0
            area = int(region.sum())
            if area == 0:
                continue
            thr, rel = config.get(pv, (1.0, True))  # unknown defect: full coverage
            if rel:
                sat = thr * area
            else:
                sat = thr * (size * size) / float(native.size)
            out.append((i, region, float(min(max(sat, 1.0), area))))
    return out


def spro(
    maps: np.ndarray,
    labels: np.ndarray,
    defects: List[Tuple[int, np.ndarray, float]],
    max_fpr: float = 0.05,
    steps: int = 200,
) -> float:
    """Area under the sPRO curve up to ``max_fpr``, normalized to [0, 1].

    The false-positive rate is measured on the anomaly-free test images
    (``labels == 0``); thresholds are their score quantiles, so the FPR axis is
    sampled evenly on ``[0, max_fpr]``. sPRO(t) = mean over defects of
    ``min(1, |{p in region: map_p >= t}| / saturation)``.

    Not the official evaluation script (that one runs on native-resolution maps);
    use ``--save-maps`` and MVTec's script for reported numbers.
    """
    if not defects or not (labels == 0).any():
        return float("nan")
    good = np.sort(maps[labels == 0].ravel())
    fprs = np.linspace(0.0, max_fpr, steps)
    # threshold with FPR ~ f: the (1 - f) quantile of the normal pixels
    ths = np.quantile(good, 1.0 - fprs)
    real_fpr = 1.0 - np.searchsorted(good, ths, side="left") / good.size
    vals = [np.sort(maps[i][region]) for i, region, _ in defects]
    sats = np.array([s for _, _, s in defects])
    curve = np.empty(steps)
    for k, t in enumerate(ths):
        hit = np.array([v.size - np.searchsorted(v, t, side="left") for v in vals])
        curve[k] = np.minimum(1.0, hit / sats).mean()
    x = np.clip(real_fpr, 0.0, max_fpr)
    trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed it
    return float(trapz(curve, x) / max_fpr)


def save_native_maps(
    maps: np.ndarray,
    entries: Sequence[TestEntry],
    out_root: Path,
    cat: str,
) -> int:
    """Write each map as float32 TIFF at the image's native resolution, in the
    layout MVTec's LOCO evaluation script reads: ``<out>/<cat>/test/<type>/<stem>.tiff``."""
    n = 0
    for amap, (p, dtype, _m) in zip(maps, entries):
        with Image.open(p) as im:
            w, h = im.size
        img = Image.fromarray(np.asarray(amap, dtype=np.float32), mode="F")
        img = img.resize((w, h), Image.BILINEAR)
        dst = Path(out_root) / cat / "test" / dtype / f"{Path(p).stem}.tiff"
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst)
        n += 1
    return n
