"""Download the pinned model weights and the few upstream files the spike needs.

Usage: uv run python tools/fetch_models.py

- Weights come from models/models.lock.json (pinned commits) and must match its sha256.
- Upstream files go to upstream/ (gitignored): the MiniFASNet model definition for the ONNX
  conversion, plus the crop code and 3 public sample images used by the tests.
This is the only step that uses the network. The app itself runs offline.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
SFAS_REPO = "minivision-ai/Silent-Face-Anti-Spoofing"
SFAS_FILES = [
    "src/model_lib/MiniFASNet.py",
    "src/generate_patches.py",
    "images/sample/image_T1.jpg",
    "images/sample/image_F1.jpg",
    "images/sample/image_F2.jpg",
]


def download(url: str, dest: Path) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return data


def main() -> int:
    lock = json.loads((MODELS / "models.lock.json").read_text())
    failed = []
    for m in lock["models"]:
        dest = MODELS / m["file"]
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == m["sha256"]:
            print(f"ok      {m['file']}")
            continue
        # The lock stores GitHub "blob" URLs; "raw" serves the file and resolves Git LFS.
        data = download(m["url"].replace("/blob/", "/raw/", 1), dest)
        if hashlib.sha256(data).hexdigest() != m["sha256"]:
            dest.unlink()
            failed.append(m["file"])
            print(f"BAD     {m['file']}: sha256 mismatch, file removed")
        else:
            print(f"fetched {m['file']} ({m['license']})")

    commit = lock["upstream_commits"][SFAS_REPO]
    for rel in SFAS_FILES:
        dest = ROOT / "upstream" / "Silent-Face-Anti-Spoofing" / rel
        if not dest.exists():
            download(f"https://raw.githubusercontent.com/{SFAS_REPO}/{commit}/{rel}", dest)
            print(f"fetched upstream/Silent-Face-Anti-Spoofing/{rel}")

    if failed:
        print(f"{len(failed)} file(s) failed the sha256 check: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
