"""Stack 1: YuNet detect -> quality gate -> MiniFASNet liveness -> SFace align + embed -> cosine match.

Usage: uv run python bench/pipeline_stack1.py IMAGE [IMAGE2]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS_DIR, Liveness, QualityGate, Result, Timer, l2_normalize  # noqa: E402

YUNET = "face_detection_yunet_2023mar.onnx"
SFACE = "face_recognition_sface_2021dec.onnx"
# OpenCV zoo reference threshold for SFace cosine similarity; tuned in T2.
COSINE_THRESHOLD = 0.363


class Stack1:
    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        min_face_px: int = 80,
        min_det_score: float = 0.8,
        detector_threshold: float = 0.6,  # faces above this count toward the "exactly one face" rule
        quality: QualityGate | None = None,
        liveness: Liveness | None = None,
        check_liveness: bool = True,
    ):
        self.min_face_px = min_face_px
        self.min_det_score = min_det_score
        self.quality = quality or QualityGate()
        self.check_liveness = check_liveness
        self.liveness = liveness or Liveness(models_dir)
        self.detector = cv2.FaceDetectorYN.create(str(models_dir / YUNET), "", (320, 320), detector_threshold)
        self.recognizer = cv2.FaceRecognizerSF.create(str(models_dir / SFACE), "")

    def embed(self, image: np.ndarray, check_liveness: bool | None = None) -> Result:
        r = Result()
        with Timer(r.timings, "detect"):
            h, w = image.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            r.reject_reason = "no_face"
            return r
        # Faces narrower than min_face_px (people far behind the user) are ignored, not counted.
        # Deviation from the handoff's literal "reject >1 face": only usable-size faces count.
        big = [f for f in faces if f[2] >= self.min_face_px]
        face = big[0] if big else max(faces, key=lambda f: f[2])
        r.det_score = float(face[14])
        r.bbox = tuple(int(round(v)) for v in face[:4])
        if not big:
            r.reject_reason = "face_too_small"
            return r
        if len(big) > 1:
            r.reject_reason = "multiple_faces"
            return r
        if r.det_score < self.min_det_score:
            r.reject_reason = "low_det_score"
            return r

        with Timer(r.timings, "quality"):
            reason = self.quality.check(image, r.bbox)
        if reason:
            r.reject_reason = reason
            return r

        if self.check_liveness if check_liveness is None else check_liveness:
            with Timer(r.timings, "liveness"):
                r.liveness_score = self.liveness.score(image, r.bbox)
            if r.liveness_score < self.liveness.threshold:
                r.reject_reason = "spoof"
                return r

        with Timer(r.timings, "embed"):
            aligned = self.recognizer.alignCrop(image, face)
            r.embedding = l2_normalize(self.recognizer.feature(aligned))
        return r

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        # Embeddings are L2-normalized, so the dot product is the cosine similarity.
        return float(np.dot(a, b))

    @staticmethod
    def is_match(similarity: float, threshold: float = COSINE_THRESHOLD) -> bool:
        return similarity >= threshold


def _describe(path: str, r: Result) -> str:
    timings = " ".join(f"{k}={v:.1f}ms" for k, v in r.timings.items())
    live = f"{r.liveness_score:.3f}" if r.liveness_score is not None else "-"
    return f"{path}: ok={r.ok} reason={r.reject_reason} det={r.det_score} live={live} {timings}"


if __name__ == "__main__":
    stack = Stack1()
    results = []
    for p in sys.argv[1:]:
        img = cv2.imread(p)
        if img is None:
            sys.exit(f"cannot read {p}")
        results.append(stack.embed(img))
        print(_describe(p, results[-1]))
    if len(results) == 2 and all(r.ok for r in results):
        sim = Stack1.similarity(results[0].embedding, results[1].embedding)
        print(f"cosine={sim:.3f} match={Stack1.is_match(sim)}")
