"""Stack 1 checks on the public upstream liveness samples (no office data).

Run: uv run --group convert --group dev pytest tests
"""

import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from common import Liveness  # noqa: E402
from pipeline_stack1 import Stack1  # noqa: E402

UPSTREAM = ROOT / "upstream" / "Silent-Face-Anti-Spoofing"
SAMPLES = UPSTREAM / "images" / "sample"
pytestmark = pytest.mark.skipif(not SAMPLES.exists(), reason="upstream clone missing")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def stack():
    return Stack1()


@pytest.mark.parametrize("name,real", [("image_T1.jpg", True), ("image_F1.jpg", False), ("image_F2.jpg", False)])
def test_liveness_matches_upstream_labels(stack, name, real):
    r = stack.embed(cv2.imread(str(SAMPLES / name)))
    assert (r.reject_reason != "spoof") == real
    assert r.ok == real


def test_same_image_similarity_is_one(stack):
    img = cv2.imread(str(SAMPLES / "image_T1.jpg"))
    a, b = stack.embed(img), stack.embed(img.copy())
    assert Stack1.similarity(a.embedding, b.embedding) == pytest.approx(1.0, abs=1e-5)


def test_rejects_frame_without_face(stack):
    assert stack.embed(np.full((480, 360, 3), 128, np.uint8)).reject_reason == "no_face"


def test_liveness_port_matches_upstream_torch(stack):
    """Our crop + ONNX must equal upstream CropImage + torch for the same bbox."""
    torch = pytest.importorskip("torch")
    crop_mod = _load(UPSTREAM / "src" / "generate_patches.py", "upstream_patches")
    net_mod = _load(UPSTREAM / "src" / "model_lib" / "MiniFASNet.py", "upstream_minifasnet")
    nets = []
    for pth, cls, scale in [("2.7_80x80_MiniFASNetV2.pth", "MiniFASNetV2", 2.7),
                            ("4_0_0_80x80_MiniFASNetV1SE.pth", "MiniFASNetV1SE", 4.0)]:
        net = getattr(net_mod, cls)(conv6_kernel=(5, 5))
        state = torch.load(ROOT / "models" / pth, map_location="cpu", weights_only=True)
        net.load_state_dict(OrderedDict((k.removeprefix("module."), v) for k, v in state.items()))
        nets.append((net.eval(), scale))

    liveness = Liveness()
    for name in ["image_T1.jpg", "image_F1.jpg", "image_F2.jpg"]:
        img = cv2.imread(str(SAMPLES / name))
        bbox = stack.embed(img).bbox
        probs = np.zeros(3)
        for net, scale in nets:
            patch = crop_mod.CropImage().crop(img, list(bbox), scale, 80, 80, True)
            x = torch.from_numpy(patch.transpose(2, 0, 1)).float()[None]
            with torch.no_grad():
                probs += torch.softmax(net(x), dim=1).numpy()[0]
        assert liveness.score(img, bbox) == pytest.approx(probs[1] / 2, abs=1e-4)


def _side_by_side(img, second_scale):
    """The sample face plus a second copy of it scaled by second_scale, on one canvas."""
    small = cv2.resize(img, None, fx=second_scale, fy=second_scale)
    canvas = np.full((img.shape[0], img.shape[1] + small.shape[1], 3), 128, np.uint8)
    canvas[:, :img.shape[1]] = img
    canvas[:small.shape[0], img.shape[1]:] = small
    return canvas


def test_small_background_face_is_ignored(stack):
    img = cv2.imread(str(SAMPLES / "image_T1.jpg"))
    canvas = _side_by_side(img, 0.25)
    stack.detector.setInputSize((canvas.shape[1], canvas.shape[0]))
    assert len(stack.detector.detect(canvas)[1]) == 2  # the detector sees both faces
    assert stack.embed(canvas).ok


def test_two_usable_faces_are_rejected(stack):
    img = cv2.imread(str(SAMPLES / "image_T1.jpg"))
    assert stack.embed(_side_by_side(img, 1.0)).reject_reason == "multiple_faces"
