"""Parts shared by Stack 1 and Stack 2: result type, quality gate, MiniFASNet liveness."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass
class Result:
    embedding: np.ndarray | None = None
    reject_reason: str | None = None
    liveness_score: float | None = None
    det_score: float | None = None
    bbox: tuple[int, int, int, int] | None = None  # x, y, w, h in input image pixels
    timings: dict[str, float] = field(default_factory=dict)  # ms per stage

    @property
    def ok(self) -> bool:
        return self.embedding is not None


class Timer:
    def __init__(self, timings: dict[str, float], stage: str):
        self.timings, self.stage = timings, stage

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        self.timings[self.stage] = (time.perf_counter() - self.t0) * 1000


@dataclass
class QualityGate:
    """Rejects blurry, dark, or overexposed faces. Thresholds are starting values for T2."""

    blur_min: float = 40.0  # Laplacian variance of the gray face crop
    brightness_min: float = 40.0  # mean gray level of the face crop
    brightness_max: float = 220.0

    def check(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> str | None:
        x, y, w, h = bbox
        h_img, w_img = image.shape[:2]
        crop = image[max(y, 0):min(y + h, h_img), max(x, 0):min(x + w, w_img)]
        if crop.size == 0:
            return "face_out_of_frame"
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        if brightness < self.brightness_min:
            return "too_dark"
        if brightness > self.brightness_max:
            return "too_bright"
        if cv2.Laplacian(gray, cv2.CV_64F).var() < self.blur_min:
            return "blurry"
        return None


def _liveness_box(src_w: int, src_h: int, bbox, scale: float) -> tuple[int, int, int, int]:
    # Port of upstream Silent-Face-Anti-Spoofing src/generate_patches.py CropImage._get_new_box.
    x, y, box_w, box_h = bbox
    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = box_w / 2 + x, box_h / 2 + y
    x1, y1 = cx - new_w / 2, cy - new_h / 2
    x2, y2 = cx + new_w / 2, cy + new_h / 2
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > src_w - 1:
        x1 -= x2 - src_w + 1
        x2 = src_w - 1
    if y2 > src_h - 1:
        y1 -= y2 - src_h + 1
        y2 = src_h - 1
    return int(x1), int(y1), int(x2), int(y2)


class Liveness:
    """MiniFASNetV2 (scale 2.7) + MiniFASNetV1SE (scale 4.0), averaged softmax.

    Score = mean probability of class 1, which upstream test.py labels "real".
    Upstream feeds 3:4 (W:H) frames and RetinaFace boxes; we feed YuNet/dlib boxes.
    """

    MODELS = [("2.7_80x80_MiniFASNetV2.onnx", 2.7), ("4_0_0_80x80_MiniFASNetV1SE.onnx", 4.0)]
    SIZE = 80
    REAL_CLASS = 1

    def __init__(self, models_dir: Path = MODELS_DIR, threshold: float = 0.5):
        self.threshold = threshold
        self.sessions = [
            (ort.InferenceSession(str(models_dir / name), providers=["CPUExecutionProvider"]), scale)
            for name, scale in self.MODELS
        ]

    def score(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        src_h, src_w = image.shape[:2]
        probs = np.zeros(3, dtype=np.float32)
        for sess, scale in self.sessions:
            x1, y1, x2, y2 = _liveness_box(src_w, src_h, bbox, scale)
            patch = cv2.resize(image[y1:y2 + 1, x1:x2 + 1], (self.SIZE, self.SIZE))
            # BGR, 0-255, no normalization: upstream removed the /255 step.
            blob = patch.transpose(2, 0, 1)[None].astype(np.float32)
            probs += sess.run(None, {"input": blob})[0][0]
        return float(probs[self.REAL_CLASS] / len(self.sessions))


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    return v / np.linalg.norm(v)
