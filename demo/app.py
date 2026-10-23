"""Local webcam demo for Stack 1: enroll -> check-in (1:1 or 1:N) -> event log.

Run: uv run python demo/app.py [--camera 0] [--reset]

Keys:
  e        enroll: type employee ID, Enter, then hold still for 5 frames (Esc cancels)
  m        switch mode: auto (1:N) / pick (1:1)
  Tab      in pick mode, select the next employee
  Space    check in (tries for up to 2 s)
  l        show/hide the event log
  q        quit

Stores only averaged embeddings in artifacts/embeddings/gallery.npz (gitignored).
No frame is written to disk. No network calls.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from common import Result, l2_normalize  # noqa: E402
from pipeline_stack1 import COSINE_THRESHOLD, Stack1  # noqa: E402

GALLERY_PATH = ROOT / "artifacts" / "embeddings" / "gallery.npz"
EVENTS_PATH = ROOT / "artifacts" / "events.csv"
ENROLL_FRAMES = 5
ENROLL_GAP_S = 0.25
CHECKIN_TIMEOUT_S = 2.0
# 1:N starting values, not tuned yet (T2 tunes them on the office set).
T_STRICT = 0.45
MARGIN = 0.05


class Gallery:
    def __init__(self, path: Path | None = None):
        self.path = path or GALLERY_PATH
        self.people: dict[str, np.ndarray] = {}
        if self.path.exists():
            data = np.load(self.path)
            self.people = {k: data[k] for k in data.files}

    def enroll(self, employee_id: str, embeddings: list[np.ndarray]) -> None:
        self.people[employee_id] = l2_normalize(np.mean(embeddings, axis=0))
        self.save()

    def remove(self, employee_id: str) -> None:
        if self.people.pop(employee_id, None) is not None:
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, **self.people)

    def rank(self, emb: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        sims = sorted(((i, float(np.dot(e, emb))) for i, e in self.people.items()), key=lambda x: -x[1])
        return sims[:k]

    def verify(self, employee_id: str, emb: np.ndarray, threshold: float = COSINE_THRESHOLD):
        """1:1. Returns (accepted, similarity)."""
        sim = float(np.dot(self.people[employee_id], emb))
        return sim >= threshold, sim

    def identify(self, emb: np.ndarray, t_strict: float = T_STRICT, margin: float = MARGIN):
        """1:N. Accept top-1 only if sim >= t_strict and top1 - top2 >= margin.

        Returns (employee_id or None, top1 similarity, reject reason or None).
        """
        if not self.people:
            return None, 0.0, "empty_gallery"
        ids = list(self.people)
        sims = np.array([float(np.dot(self.people[i], emb)) for i in ids])
        order = np.argsort(sims)[::-1]
        top1 = sims[order[0]]
        top2 = sims[order[1]] if len(ids) > 1 else -1.0
        if top1 < t_strict:
            return None, top1, "unknown"
        if top1 - top2 < margin:
            return None, top1, "ambiguous"
        return ids[order[0]], top1, None


@dataclass
class Event:
    employee_id: str
    timestamp: str
    mode: str
    similarity: float | None
    liveness_score: float | None
    result: str
    reject_reason: str | None


def to_portrait_3x4(frame: np.ndarray) -> np.ndarray:
    """Center-crop to W:H = 3:4. MiniFASNet upstream was trained on 3:4 frames."""
    h, w = frame.shape[:2]
    target_w = int(h * 3 / 4)
    if target_w <= w:
        x0 = (w - target_w) // 2
        return frame[:, x0:x0 + target_w]
    target_h = int(w * 4 / 3)
    y0 = (h - target_h) // 2
    return frame[y0:y0 + target_h]


def checkin(gallery: Gallery, r: Result, mode: str, selected: str | None) -> Event:
    now = datetime.now().isoformat(timespec="seconds")
    if not r.ok:
        return Event(selected or "-", now, mode, None, r.liveness_score, "FAIL", r.reject_reason)
    if mode == "pick":
        ok, sim = gallery.verify(selected, r.embedding)
        return Event(selected, now, mode, sim, r.liveness_score, "PASS" if ok else "FAIL",
                     None if ok else "below_threshold")
    who, sim, reason = gallery.identify(r.embedding)
    return Event(who or "-", now, mode, sim, r.liveness_score, "PASS" if who else "FAIL", reason)


def append_event(ev: Event) -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not EVENTS_PATH.exists()
    with EVENTS_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(ev)))
        if new:
            w.writeheader()
        w.writerow(asdict(ev))


def _text(img, s, y, color=(255, 255, 255), scale=0.55):
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def run(camera: int) -> None:
    stack = Stack1()
    gallery = Gallery()
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        sys.exit("Cannot open camera. Allow camera access for this terminal in System Settings > Privacy.")

    mode, selected_idx = "auto", 0
    state, typed = "idle", ""  # idle | typing | enrolling | checkin
    captured: list[np.ndarray] = []
    last_capture = deadline = 0.0
    message, msg_color = "Press e to enroll, Space to check in", (255, 255, 255)
    log: deque[Event] = deque(maxlen=50)
    show_log = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = to_portrait_3x4(cv2.flip(frame, 1))
        r = stack.embed(frame)
        view = frame.copy()
        ids = sorted(gallery.people)
        selected = ids[selected_idx % len(ids)] if ids else None

        if r.bbox:
            x, y, w, h = r.bbox
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 200, 0) if r.ok else (0, 0, 255), 2)
        live = f"{r.liveness_score:.2f}" if r.liveness_score is not None else "-"
        total_ms = sum(r.timings.values())
        _text(view, f"live={live} reason={r.reject_reason or 'ok'} {total_ms:.0f}ms", 20)
        pick = f" -> {selected}" if mode == "pick" else ""
        _text(view, f"mode={mode}{pick}  enrolled={len(ids)}", 42)

        now = time.monotonic()
        if state == "typing":
            message, msg_color = f"Employee ID: {typed}_", (255, 255, 0)
        elif state == "enrolling":
            if r.ok and now - last_capture >= ENROLL_GAP_S:
                captured.append(r.embedding)
                last_capture = now
            message, msg_color = f"Enrolling {typed}: {len(captured)}/{ENROLL_FRAMES}", (255, 255, 0)
            if len(captured) >= ENROLL_FRAMES:
                gallery.enroll(typed, captured)
                message, msg_color = f"Enrolled {typed}", (0, 255, 0)
                state = "idle"
        elif state == "checkin" and (r.ok or now >= deadline):
            ev = checkin(gallery, r, mode, selected)
            log.appendleft(ev)
            append_event(ev)
            sim = f"{ev.similarity:.3f}" if ev.similarity is not None else "-"
            message = f"{ev.result} {ev.employee_id} sim={sim} reason={ev.reject_reason or '-'}"
            msg_color = (0, 255, 0) if ev.result == "PASS" else (0, 0, 255)
            state = "idle"

        _text(view, message, view.shape[0] - 15, msg_color, 0.6)
        if show_log:
            for i, ev in enumerate(list(log)[:12]):
                sim = f"{ev.similarity:.2f}" if ev.similarity is not None else "-"
                _text(view, f"{ev.timestamp[11:]} {ev.mode} {ev.employee_id} {ev.result} s={sim} "
                            f"l={ev.liveness_score or 0:.2f} {ev.reject_reason or ''}", 70 + i * 20, scale=0.45)
        cv2.imshow("Face Attendance Spike (Stack 1)", view)

        key = cv2.waitKey(1) & 0xFF
        if state == "typing":
            if key == 27:
                state, message = "idle", "Enroll cancelled"
            elif key in (13, 10) and typed:
                state, captured, last_capture = "enrolling", [], 0.0
            elif key in (8, 127):
                typed = typed[:-1]
            elif 32 < key < 127:
                typed += chr(key)
            continue
        if key == ord("q"):
            break
        if key == 27 and state == "enrolling":
            state, message = "idle", "Enroll cancelled"
        elif key == ord("e"):
            state, typed = "typing", ""
        elif key == ord("m"):
            mode = "pick" if mode == "auto" else "auto"
        elif key == 9:
            selected_idx += 1
        elif key == ord("l"):
            show_log = not show_log
        elif key == ord(" ") and state == "idle":
            if not ids:
                message, msg_color = "Enroll someone first (e)", (0, 0, 255)
            else:
                state, deadline = "checkin", now + CHECKIN_TIMEOUT_S

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--reset", action="store_true", help="delete the local gallery before start")
    args = ap.parse_args()
    if args.reset and GALLERY_PATH.exists():
        GALLERY_PATH.unlink()
    run(args.camera)
