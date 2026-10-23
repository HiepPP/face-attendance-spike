"""Local web app for Stack 1 testing. Serves on 127.0.0.1 only; frames never leave this machine.

Run: uv run python demo/web.py [--port 8765] [--data-dir data/demo] [--reset]
Then open http://127.0.0.1:8765 in Chrome or Safari.

Uploaded enrollment photos are kept in --data-dir (see photo_store.py). The gallery is rebuilt
from that folder at startup. Webcam enrollments and check-in frames are never written to disk.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from collections import deque
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app  # noqa: E402
from app import Event, Gallery, append_event, to_portrait_3x4  # noqa: E402
from photo_store import CONDITIONS, PhotoStore, valid_id  # noqa: E402
from pipeline_stack1 import COSINE_THRESHOLD, Stack1  # noqa: E402

INDEX = Path(__file__).resolve().parent / "static" / "index.html"
DEFAULT_DATA_DIR = app.ROOT / "data" / "demo"
MAX_BODY = 40 * 1024 * 1024
# Uploaded photos are downscaled for detection only; YuNet is trained on faces of about 10-300 px.
# Scale by the short side, like the 480x640 webcam frames, so tall phone screenshots keep enough face pixels.
UPLOAD_SHORT_SIDE = 480


def fit(img: np.ndarray, short_side: int = UPLOAD_SHORT_SIDE) -> np.ndarray:
    s = short_side / min(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def decode(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64.split(",", 1)[-1])
    # IMREAD_COLOR applies the EXIF orientation, so phone photos come out upright.
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode image (use JPEG, PNG, or WebP; HEIC is not supported)")
    return img


class Service:
    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self.stack = Stack1()
        self.gallery = Gallery(app.GALLERY_PATH)
        self.store = PhotoStore(data_dir)
        self.log: deque[Event] = deque(maxlen=50)
        self.lock = threading.RLock()  # OpenCV/ORT objects are shared; serialize pipeline calls
        self.photo_emb: dict[Path, np.ndarray | None] = {}
        self.rejected: dict[str, list[str]] = {}
        with self.lock:
            for emp in self.store.people():
                self._rebuild(emp)

    # --- pipeline helpers -------------------------------------------------------------------
    def _embed(self, img: np.ndarray, liveness_threshold=None, check_liveness=None):
        with self.lock:
            if liveness_threshold is not None:
                self.stack.liveness.threshold = liveness_threshold
            return self.stack.embed(img, check_liveness)

    def _photo_embedding(self, photo: Path) -> np.ndarray | None:
        if photo not in self.photo_emb:
            img = cv2.imread(str(photo))
            r = self.stack.embed(fit(img), check_liveness=False) if img is not None else None
            self.photo_emb[photo] = r.embedding if r and r.ok else None
        return self.photo_emb[photo]

    def _rebuild(self, emp: str) -> None:
        """Template = mean of the 'normal' photos, or of all photos if there is no 'normal' photo."""
        photos = self.store.photos(emp)
        normal = [p for p in photos if self.store.condition(p) == "normal"] or photos
        embs = [e for e in (self._photo_embedding(p) for p in normal) if e is not None]
        self.rejected[emp] = [p.name for p in photos if self._photo_embedding(p) is None]
        if embs:
            self.gallery.enroll(emp, embs)
        else:
            self.gallery.remove(emp)

    @staticmethod
    def describe(r) -> dict:
        return {"ok": r.ok, "reason": r.reject_reason, "bbox": r.bbox, "det_score": r.det_score,
                "liveness": r.liveness_score, "timings": r.timings}

    def _match(self, r, mode: str, emp: str | None, th: dict, source: str) -> Event:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        if not r.ok:
            ev = Event(emp or "-", now, source, None, r.liveness_score, "FAIL", r.reject_reason)
        elif mode == "pick":
            accepted, sim = self.gallery.verify(emp, r.embedding, th.get("verify", COSINE_THRESHOLD))
            ev = Event(emp, now, source, sim, r.liveness_score, "PASS" if accepted else "FAIL",
                       None if accepted else "below_threshold")
        else:
            who, sim, reason = self.gallery.identify(
                r.embedding, th.get("t_strict", app.T_STRICT), th.get("margin", app.MARGIN))
            ev = Event(who or "-", now, source, sim, r.liveness_score, "PASS" if who else "FAIL", reason)
        self.log.appendleft(ev)
        append_event(ev)
        return ev

    # --- API --------------------------------------------------------------------------------
    def analyze(self, body: dict) -> dict:
        """Live preview: pipeline result plus a 1:N match for the label. Nothing is logged."""
        img = to_portrait_3x4(decode(body["frame"]))
        r = self._embed(img, body.get("liveness_threshold"))
        out = self.describe(r)
        if r.ok and self.gallery.people:
            th = body.get("thresholds", {})
            who, sim, reason = self.gallery.identify(
                r.embedding, th.get("t_strict", app.T_STRICT), th.get("margin", app.MARGIN))
            out["match"] = {"employee_id": who, "similarity": sim, "reason": reason}
        return out

    def enroll(self, body: dict) -> dict:
        """Webcam enroll: embeddings only, frames are not saved."""
        emp = body["employee_id"].strip()
        if not valid_id(emp):
            return {"ok": False, "reason": "invalid_employee_id (A-Z a-z 0-9 _ -)"}
        results = [self._embed(to_portrait_3x4(decode(f)), body.get("liveness_threshold")) for f in body["frames"]]
        good = [r.embedding for r in results if r.ok]
        if len(good) < 3:
            reasons = sorted({r.reject_reason for r in results if not r.ok})
            return {"ok": False, "accepted": len(good), "reason": ",".join(reasons)}
        with self.lock:
            self.gallery.enroll(emp, good)
            if body.get("name") is not None:
                self.store.set_name(emp, body["name"])
        return {"ok": True, "accepted": len(good), "employee_id": emp}

    def upload(self, body: dict) -> dict:
        """Upload enroll: one photo per call. Accepted photos are saved, then the template is rebuilt."""
        emp, condition = body["employee_id"].strip(), body.get("condition", "normal")
        if not valid_id(emp):
            return {"ok": False, "reason": "invalid_employee_id (A-Z a-z 0-9 _ -)"}
        if condition not in CONDITIONS:
            return {"ok": False, "reason": "invalid_condition"}
        img = decode(body["file"])
        # Liveness is off: an enrollment photo is not a presentation attack test.
        r = self._embed(fit(img), check_liveness=False)
        if not r.ok:
            return {"ok": False, "reason": r.reject_reason}
        with self.lock:
            path = self.store.save(emp, condition, img)
            self.photo_emb[path] = r.embedding
            if body.get("name") is not None:
                self.store.set_name(emp, body["name"])
            self._rebuild(emp)
        return {"ok": True, "employee_id": emp, "saved": path.name, "photos": len(self.store.photos(emp))}

    def checkin(self, body: dict) -> dict:
        """Webcam check-in: one attempt. The client retries until ok or 2 s, then sends final=true."""
        r = self._embed(to_portrait_3x4(decode(body["frame"])), body.get("liveness_threshold"))
        if not r.ok and not body.get("final"):
            return {"done": False, **self.describe(r)}
        mode, emp = body["mode"], body.get("employee_id")
        if mode == "pick" and emp not in self.gallery.people:
            return {"done": True, "event": None, "reason": "unknown_employee"}
        ev = self._match(r, mode, emp, body.get("thresholds", {}), mode)
        return {"done": True, "event": asdict(ev), **self.describe(r)}

    def recognize(self, body: dict) -> dict:
        """Check-in from an uploaded photo. The photo is not saved."""
        mode, emp = body["mode"], body.get("employee_id")
        if mode == "pick" and emp not in self.gallery.people:
            return {"event": None, "reason": "unknown_employee"}
        img = fit(decode(body["file"]))
        r = self._embed(img, body.get("liveness_threshold"), bool(body.get("check_liveness", True)))
        ev = self._match(r, mode, emp, body.get("thresholds", {}), f"{mode}/upload")
        top = self.gallery.rank(r.embedding) if r.ok else []
        return {"event": asdict(ev), "top": top, "size": [img.shape[1], img.shape[0]], **self.describe(r)}

    def rename(self, body: dict) -> dict:
        with self.lock:
            self.store.set_name(body["employee_id"], body.get("name", ""))
        return {"ok": True}

    def delete(self, body: dict) -> dict:
        emp = body["employee_id"]
        with self.lock:
            self.gallery.remove(emp)
            if valid_id(emp):
                self.store.remove(emp)
        return {"ok": True}

    def state(self) -> dict:
        ids = sorted(set(self.gallery.people) | set(self.store.people()))
        return {
            "employees": [{"id": i, "name": self.store.names.get(i, ""), "photos": len(self.store.photos(i)) if valid_id(i) else 0,
                           "enrolled": i in self.gallery.people, "rejected": self.rejected.get(i, [])} for i in ids],
            "log": [asdict(e) for e in self.log],
            "conditions": CONDITIONS,
            "data_dir": str(self.store.root),
            "defaults": {"verify": COSINE_THRESHOLD, "t_strict": app.T_STRICT, "margin": app.MARGIN,
                         "liveness": self.stack.liveness.threshold},
        }


def make_handler(service: Service):
    routes = {"/api/analyze": service.analyze, "/api/enroll": service.enroll, "/api/upload": service.upload,
              "/api/checkin": service.checkin, "/api/recognize": service.recognize,
              "/api/rename": service.rename, "/api/delete": service.delete}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: bytes, ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/":
                self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._send(200, json.dumps(service.state(), ensure_ascii=False).encode())
            else:
                self._send(404, b"{}")

        def do_POST(self):
            fn = routes.get(self.path)
            length = int(self.headers.get("Content-Length", 0))
            if fn is None or length > MAX_BODY:
                self._send(404 if fn is None else 413, b"{}")
                return
            try:
                out = fn(json.loads(self.rfile.read(length)))
                self._send(200, json.dumps(out, ensure_ascii=False).encode())
            except (KeyError, ValueError) as e:
                self._send(400, json.dumps({"error": str(e)}).encode())

        def log_message(self, fmt, *args):  # keep the console quiet; no request bodies are logged
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="folder for uploaded enrollment photos")
    ap.add_argument("--reset", action="store_true", help="delete the embedding gallery before start (photos stay)")
    args = ap.parse_args()
    if args.reset and app.GALLERY_PATH.exists():
        app.GALLERY_PATH.unlink()
    service = Service(args.data_dir.resolve())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(service))
    print(f"Photos: {service.store.root}  Employees: {len(service.gallery.people)}")
    print(f"Open http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
