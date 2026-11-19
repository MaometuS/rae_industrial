"""MVTec LOCO loader and sPRO on a synthetic tree (numpy + PIL only, no GPU)."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from iad import datasets as D

H, W = 40, 60  # non-square native size, like most LOCO categories


def _png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _rgb() -> np.ndarray:
    return np.full((H, W, 3), 128, np.uint8)


@pytest.fixture
def loco(tmp_path: Path) -> Path:
    c = tmp_path / "pushpins"
    for i in range(3):
        _png(c / "train" / "good" / f"{i:03d}.png", _rgb())
    for i in range(2):
        _png(c / "validation" / "good" / f"{i:03d}.png", _rgb())
    for i in range(2):
        _png(c / "test" / "good" / f"{i:03d}.png", _rgb())
    _png(c / "test" / "logical_anomalies" / "000.png", _rgb())
    _png(c / "test" / "structural_anomalies" / "000.png", _rgb())
    # logical image: two defects (value 255 = missing pin, 200 = extra pin)
    a = np.zeros((H, W)); a[0:10, 0:10] = 255
    b = np.zeros((H, W)); b[20:30, 30:40] = 200
    _png(c / "ground_truth" / "logical_anomalies" / "000" / "000.png", a)
    _png(c / "ground_truth" / "logical_anomalies" / "000" / "001.png", b)
    s = np.zeros((H, W)); s[30:40, 0:20] = 100
    _png(c / "ground_truth" / "structural_anomalies" / "000" / "000.png", s)
    (c / "defects_config.json").write_text(json.dumps([
        {"defect_name": "missing", "pixel_value": 255, "saturation_threshold": 0.5,
         "relative_saturation": True},
        {"defect_name": "extra", "pixel_value": 200, "saturation_threshold": 50,
         "relative_saturation": False},
        {"defect_name": "scratch", "pixel_value": 100, "saturation_threshold": 1.0,
         "relative_saturation": True},
    ]))
    return tmp_path


def test_loco_split(loco: Path):
    assert D.is_loco(loco, "pushpins")
    train, val, test = D.split(loco, "pushpins")
    assert len(train) == 3 and len(val) == 2       # official val, train untouched
    types = [d for _p, d, _m in test]
    assert types.count("good") == 2
    assert {"logical_anomalies", "structural_anomalies"} <= set(types)
    for _p, d, m in test:
        assert (m is None) == (d == "good")
        if d == "logical_anomalies":
            assert isinstance(m, list) and len(m) == 2


def test_mvtec_ad_layout_unchanged(tmp_path: Path):
    c = tmp_path / "carpet"
    _png(c / "train" / "good" / "000.png", _rgb())
    _png(c / "test" / "good" / "000.png", _rgb())
    _png(c / "test" / "hole" / "000.png", _rgb())
    m = np.zeros((H, W)); m[:5, :5] = 255
    _png(c / "ground_truth" / "hole" / "000_mask.png", m)
    assert not D.is_loco(tmp_path, "carpet")
    train, val, test = D.split(tmp_path, "carpet")
    assert val is None and len(train) == 1
    hole = [e for e in test if e[1] == "hole"][0]
    assert isinstance(hole[2], Path) and D.load_mask_any(hole[2], 16).any()


def test_merged_mask_and_defects(loco: Path):
    _t, _v, test = D.split(loco, "pushpins")
    size = 32
    logical = [m for _p, d, m in test if d == "logical_anomalies"][0]
    merged = D.load_mask_any(logical, size)
    assert merged.any()
    cfg = D.loco_defects_config(loco, "pushpins")
    assert cfg[255] == (0.5, True) and cfg[200] == (50.0, False)
    defs = D.loco_defects([m for _p, _d, m in test], size, cfg)
    assert len(defs) == 3                                  # 2 logical + 1 structural
    for _i, region, sat in defs:
        assert 1.0 <= sat <= region.sum()                  # capped at the region area
    # the union of the per-defect regions is the merged mask of that image
    idx = [i for i, (_p, d, _m) in enumerate(test) if d == "logical_anomalies"][0]
    union = np.zeros((size, size), bool)
    for i, region, _s in defs:
        if i == idx:
            union |= region
    assert (union == merged).all()


def test_spro_perfect_and_random(loco: Path):
    _t, _v, test = D.split(loco, "pushpins")
    size = 32
    labels = np.array([int(d != "good") for _p, d, _m in test])
    masks = np.stack([D.load_mask_any(m, size) for _p, _d, m in test])
    defs = D.loco_defects([m for _p, _d, m in test], size, D.loco_defects_config(loco, "pushpins"))
    perfect = masks.astype(np.float32) + 1e-3 * np.random.default_rng(0).random(masks.shape)
    assert D.spro(perfect, labels, defs) > 0.95
    rnd = np.random.default_rng(1).random(masks.shape).astype(np.float32)
    assert D.spro(rnd, labels, defs) < 0.2
    assert np.isnan(D.spro(rnd, labels, []))


def test_save_native_maps(loco: Path, tmp_path: Path):
    _t, _v, test = D.split(loco, "pushpins")
    maps = np.random.default_rng(0).random((len(test), 32, 32)).astype(np.float32)
    n = D.save_native_maps(maps, test, tmp_path / "maps", "pushpins")
    assert n == len(test)
    f = tmp_path / "maps" / "pushpins" / "test" / "logical_anomalies" / "000.tiff"
    with Image.open(f) as im:
        assert im.size == (W, H) and im.mode == "F"
