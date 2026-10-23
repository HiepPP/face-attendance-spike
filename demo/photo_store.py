"""Enrollment photos on disk, in the T2 benchmark layout.

<root>/<employee_id>/<condition>_<n>.jpg   photos (re-encoded JPEG, EXIF stripped)
<root>/people.json                         {employee_id: display name}
<root>/_deleted/<employee_id>_<timestamp>/ removed employees (moved, not erased)

The root is under data/, which is gitignored and blocked by the pre-commit hook.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import cv2
import numpy as np

CONDITIONS = ("normal", "backlight", "glasses", "mask", "angle", "morning", "evening")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
PHOTO_RE = re.compile(r"^([a-z]+)_(\d+)\.jpg$")


def valid_id(employee_id: str) -> bool:
    # The ID becomes a folder name, so only allow characters that cannot escape the root.
    return bool(ID_RE.match(employee_id))


class PhotoStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.names_path = root / "people.json"
        self.names: dict[str, str] = json.loads(self.names_path.read_text()) if self.names_path.exists() else {}

    def _dir(self, employee_id: str) -> Path:
        if not valid_id(employee_id):
            raise ValueError("invalid employee_id")
        return self.root / employee_id

    def set_name(self, employee_id: str, name: str) -> None:
        self._dir(employee_id)
        name = name.strip()
        if name:
            self.names[employee_id] = name[:100]
        else:
            self.names.pop(employee_id, None)
        self.names_path.write_text(json.dumps(self.names, ensure_ascii=False, indent=2))

    def people(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and valid_id(p.name) and self.photos(p.name))

    def photos(self, employee_id: str) -> list[Path]:
        d = self._dir(employee_id)
        return sorted(p for p in d.glob("*.jpg") if PHOTO_RE.match(p.name)) if d.is_dir() else []

    @staticmethod
    def condition(photo: Path) -> str:
        return PHOTO_RE.match(photo.name).group(1)

    def save(self, employee_id: str, condition: str, image: np.ndarray) -> Path:
        if condition not in CONDITIONS:
            raise ValueError("invalid condition")
        d = self._dir(employee_id)
        d.mkdir(exist_ok=True)
        used = [int(PHOTO_RE.match(p.name).group(2)) for p in d.glob(f"{condition}_*.jpg") if PHOTO_RE.match(p.name)]
        path = d / f"{condition}_{max(used, default=0) + 1}.jpg"
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError("cannot encode image")
        path.write_bytes(buf.tobytes())
        return path

    def remove(self, employee_id: str) -> None:
        d = self._dir(employee_id)
        if d.is_dir():
            trash = self.root / "_deleted"
            trash.mkdir(exist_ok=True)
            d.rename(trash / f"{employee_id}_{time.strftime('%Y%m%d-%H%M%S')}")
        if employee_id in self.names:
            self.set_name(employee_id, "")
