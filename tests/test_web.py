"""Upload enroll, persistence, recognition, rename and delete through the web Service (no HTTP, no camera)."""

import base64
from collections import deque
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
import app  # noqa: E402
import web  # noqa: E402
from photo_store import PhotoStore  # noqa: E402

SAMPLES = ROOT / "upstream" / "Silent-Face-Anti-Spoofing" / "images" / "sample"
pytestmark = pytest.mark.skipif(not SAMPLES.exists(), reason="upstream clone missing")


def b64(img: np.ndarray) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode()


@pytest.fixture
def svc(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "GALLERY_PATH", tmp_path / "gallery.npz")
    monkeypatch.setattr(app, "EVENTS_PATH", tmp_path / "events.csv")
    return web.Service(tmp_path / "photos")


@pytest.fixture
def real():
    return cv2.imread(str(SAMPLES / "image_T1.jpg"))


def test_upload_saves_photo_and_enrolls(svc, real):
    r = svc.upload({"employee_id": "NV001", "name": "Nguyễn Văn A", "condition": "normal", "file": b64(real)})
    assert r["ok"] and r["saved"] == "normal_1.jpg"
    assert (svc.store.root / "NV001" / "normal_1.jpg").exists()
    assert "NV001" in svc.gallery.people
    emp = svc.state()["employees"][0]
    assert (emp["id"], emp["name"], emp["photos"]) == ("NV001", "Nguyễn Văn A", 1)


def test_upload_rejects_bad_input_without_saving(svc):
    blank = np.full((640, 480, 3), 128, np.uint8)
    assert svc.upload({"employee_id": "NV002", "file": b64(blank)})["reason"] == "no_face"
    assert svc.upload({"employee_id": "../evil", "file": b64(blank)})["reason"].startswith("invalid_employee_id")
    assert not (svc.store.root / "NV002").exists() and svc.store.people() == []


def test_large_photo_is_downscaled_for_detection(svc, real):
    big = cv2.resize(real, None, fx=6, fy=6)  # 2880x3840, face far above YuNet's trained size
    assert svc.upload({"employee_id": "NV003", "file": b64(big)})["ok"]


def test_recognize_upload_identifies_and_blocks_spoof(svc, real):
    svc.upload({"employee_id": "NV001", "file": b64(real)})
    r = svc.recognize({"mode": "auto", "file": b64(real)})
    assert r["event"]["result"] == "PASS" and r["event"]["employee_id"] == "NV001"
    assert r["top"][0][0] == "NV001"
    spoof = cv2.imread(str(SAMPLES / "image_F2.jpg"))
    r = svc.recognize({"mode": "auto", "file": b64(spoof), "check_liveness": True})
    assert r["event"]["reject_reason"] == "spoof"
    assert not (svc.store.root / "NV001" / "normal_2.jpg").exists()  # recognition never saves


def test_gallery_is_rebuilt_from_folder_after_restart(svc, real, tmp_path):
    svc.upload({"employee_id": "NV001", "name": "A", "file": b64(real)})
    app.GALLERY_PATH.unlink()
    again = web.Service(tmp_path / "photos")
    assert "NV001" in again.gallery.people and again.store.names["NV001"] == "A"


def test_rename_and_delete_moves_folder(svc, real):
    svc.upload({"employee_id": "NV001", "file": b64(real)})
    svc.rename({"employee_id": "NV001", "name": "Trần B"})
    assert svc.state()["employees"][0]["name"] == "Trần B"
    svc.delete({"employee_id": "NV001"})
    assert svc.state()["employees"] == []
    assert len(list((svc.store.root / "_deleted").iterdir())) == 1


def test_photo_numbering_per_condition(tmp_path, real):
    store = PhotoStore(tmp_path)
    names = [store.save("NV1", c, real).name for c in ("normal", "normal", "mask")]
    assert names == ["normal_1.jpg", "normal_2.jpg", "mask_1.jpg"]
    with pytest.raises(ValueError):
        store.save("NV1", "sunny", real)


def test_live_analyze_labels_recognized_employee(svc, real):
    assert "match" not in svc.analyze({"frame": b64(real)})  # empty gallery: no label
    svc.upload({"employee_id": "NV001", "name": "A", "file": b64(real)})
    m = svc.analyze({"frame": b64(real)})["match"]
    assert m["employee_id"] == "NV001" and m["similarity"] > 0.9
    assert svc.log == deque()  # live preview does not create check-in events
