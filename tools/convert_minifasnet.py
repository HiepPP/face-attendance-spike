"""Convert the two MiniFASNet .pth weights to ONNX (dev-only, needs the `convert` group).

Usage:
    uv run --group convert python -I tools/convert_minifasnet.py \
        upstream/Silent-Face-Anti-Spoofing/src/model_lib/MiniFASNet.py models

The exported graph includes the softmax, so the output is class probabilities
[fake_0, real, fake_2]. Upstream test.py treats class index 1 as "real".
Input is BGR float in range 0-255 (upstream removed the /255 step), NCHW 1x3x80x80.
"""

import hashlib
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

# (weight file, model class, upstream crop scale), as parsed by upstream src/utility.py
MODELS = [
    ("2.7_80x80_MiniFASNetV2.pth", "MiniFASNetV2", 2.7),
    ("4_0_0_80x80_MiniFASNetV1SE.pth", "MiniFASNetV1SE", 4.0),
]
INPUT_SIZE = 80
OPSET = 17


def load_model_lib(path: Path):
    spec = importlib.util.spec_from_file_location("minifasnet_upstream", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class WithSoftmax(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(model_lib: Path, models_dir: Path) -> None:
    lib = load_model_lib(model_lib)
    # Same kernel rule as upstream get_kernel(80, 80).
    kernel = ((INPUT_SIZE + 15) // 16, (INPUT_SIZE + 15) // 16)
    rng = np.random.default_rng(0)
    derived = []
    for pth_name, cls_name, scale in MODELS:
        pth = models_dir / pth_name
        net = getattr(lib, cls_name)(conv6_kernel=kernel)
        state = torch.load(pth, map_location="cpu", weights_only=True)
        # Upstream weights were saved from DataParallel; strip the "module." prefix.
        state = OrderedDict((k.removeprefix("module."), v) for k, v in state.items())
        net.load_state_dict(state)
        model = WithSoftmax(net).eval()

        onnx_path = models_dir / (Path(pth_name).stem + ".onnx")
        dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
        torch.onnx.export(
            model, dummy, onnx_path, input_names=["input"], output_names=["prob"],
            opset_version=OPSET, dynamo=False,
        )

        x = rng.uniform(0, 255, size=(4, 3, INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        onnx_out = np.concatenate([sess.run(None, {"input": x[i:i + 1]})[0] for i in range(len(x))])
        with torch.no_grad():
            torch_out = model(torch.from_numpy(x)).numpy()
        max_diff = float(np.abs(onnx_out - torch_out).max())
        assert max_diff < 1e-4, f"{onnx_path.name}: ONNX/torch mismatch {max_diff}"

        derived.append({
            "file": onnx_path.name,
            "derived_from": pth_name,
            "sha256_source": sha256(pth),
            "sha256": sha256(onnx_path),
            "scale": scale,
            "input": "input: float32 [1,3,80,80], BGR, range 0-255",
            "output": "prob: float32 [1,3], softmax, index 1 = real",
            "opset": OPSET,
            "torch": torch.__version__,
            "max_abs_diff_vs_torch": max_diff,
        })
        print(f"{onnx_path.name} sha256={derived[-1]['sha256']} max_diff={max_diff:.2e}")

    lock_path = models_dir / "models.lock.json"
    lock = json.loads(lock_path.read_text())
    lock["derived"] = derived
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
