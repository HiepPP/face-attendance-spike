"""Demo logic checks without a camera."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))
from app import Gallery, to_portrait_3x4  # noqa: E402


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_identify_accepts_clear_top1_and_rejects_unknown_and_ambiguous(tmp_path):
    g = Gallery(tmp_path / "g.npz")
    g.enroll("a", [_unit([1, 0, 0])])
    g.enroll("b", [_unit([0, 1, 0])])
    assert g.identify(_unit([1, 0.1, 0]))[0] == "a"
    assert g.identify(_unit([0, 0, 1]))[2] == "unknown"
    assert g.identify(_unit([1, 1, 0]))[2] == "ambiguous"


def test_verify_and_persistence(tmp_path):
    g = Gallery(tmp_path / "g.npz")
    g.enroll("a", [_unit([1, 0, 0]), _unit([0.9, 0.1, 0])])
    reloaded = Gallery(tmp_path / "g.npz")
    assert reloaded.verify("a", _unit([1, 0, 0]))[0]
    assert not reloaded.verify("a", _unit([0, 1, 0]))[0]


def test_portrait_crop_is_3x4():
    assert to_portrait_3x4(np.zeros((720, 1280, 3), np.uint8)).shape[:2] == (720, 540)
    assert to_portrait_3x4(np.zeros((1000, 600, 3), np.uint8)).shape[:2] == (800, 600)
