"""Export the three pinned checkpoints to ONNX for in-browser inference (onnxruntime-web).

Usage: PYTHONPATH=src python deploy/export_onnx.py [out_dir]
Verifies each source checkpoint against config/task12.toml pins, exports static-shape
graphs, checks ONNX Runtime against PyTorch, and writes manifest.json for the browser.
"""

import hashlib
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from damage_analytics.task07 import checkpoints as crack_ck
from damage_analytics.task07.data import MEAN, STD, prepare_image
from damage_analytics.task09.checkpoints import load as load_seg
from damage_analytics.task09.config import FeaturePolicy  # noqa: F401  (import check)
from damage_analytics.task10 import checkpoints as sev_ck
from damage_analytics.task10.config import CLASS_NAMES as SEVERITY_NAMES
from damage_analytics.training.transforms import prepare_pair

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "frontend/public/models")
OUT.mkdir(parents=True, exist_ok=True)
for old in OUT.glob("*.onnx"):
    old.unlink()
cfg_path = Path("config/task12.toml")
cfg = tomllib.loads(cfg_path.read_text())
cpu = torch.device("cpu")


def source(role: str) -> tuple[Path, str]:
    path = (cfg_path.parent / cfg[role]["path"]).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != cfg[role]["sha256"]:
        raise SystemExit(f"{role}: checkpoint does not match its pin")
    return path, digest


def export(model, name, example, output_names):
    target = OUT / f"{name}.onnx"
    torch.onnx.export(model, example, str(target), input_names=["input"],
                      output_names=output_names, opset_version=17, dynamo=False)
    return target


manifest: dict = {"format": 1}
rng = np.random.default_rng(1)
test = Image.fromarray((rng.random((300, 420, 3)) * 255).astype(np.uint8))

# --- segmentation -----------------------------------------------------------------------
path, digest = source("segmentation")
payload = torch.load(path, map_location="cpu", weights_only=True)
meta = payload["metadata"]
model, payload, data = load_seg(path, cpu, meta["dataset"])
size_hw = tuple(data.image_size)
example = torch.zeros(1, 3, *size_hw)
file = export(model, "segmentation", example, ["logits"])  # 1 x 9 x H x W class scores
rgb, _, _ = prepare_pair(test, Image.new("L", test.size), data, training=False,
                         rng=np.random.default_rng(42))
with torch.inference_mode():
    expected = model(torch.from_numpy(rgb).unsqueeze(0))[0].argmax(0).numpy()
got = ort.InferenceSession(str(file)).run(None, {"input": rgb[None]})[0][0].argmax(0)
agree = float((got == expected).mean())
print(f"segmentation  input {size_hw}  ONNX/PyTorch pixel agreement {agree:.6f}")
assert agree > 0.999, "segmentation ONNX diverges from PyTorch"
manifest["segmentation"] = {
    "file": "segmentation.onnx", "sha256": digest, "epoch": payload["epoch"],
    "dataset": meta["dataset"], "height": size_hw[0], "width": size_hw[1],
    "mean": list(data.mean), "std": list(data.std), "classes": 9,
}

# --- crack classifier -------------------------------------------------------------------
path, digest = source("crack")
ck = crack_ck.read_payload(path)
cmodel, ck = crack_ck.load(path, cpu, ck)
size = int(ck["settings"]["image_size"])
file = export(cmodel, "crack", torch.zeros(1, 3, size, size), ["logits"])
x = prepare_image(test, size, training=False, rng=np.random.default_rng(0))
with torch.inference_mode():
    expected = cmodel(torch.from_numpy(x).unsqueeze(0)).softmax(1)[0].numpy()
got = ort.InferenceSession(str(file)).run(None, {"input": x[None]})[0][0]
got = np.exp(got - got.max()); got /= got.sum()
print(f"crack         softmax max abs diff {np.abs(got - expected).max():.2e}")
assert np.abs(got - expected).max() < 1e-3
manifest["crack"] = {
    "file": "crack.onnx", "sha256": digest, "epoch": ck["epoch"], "size": size,
    "threshold": ck["threshold"]["threshold"], "dataset": ck["dataset"],
    "mean": list(MEAN), "std": list(STD),
}

# --- severity classifier ----------------------------------------------------------------
path, digest = source("severity")
sp = torch.load(path, map_location="cpu", weights_only=True)
smodel, sp = sev_ck.load(path, cpu, sp["metadata"]["dataset"])
size = int(sp["metadata"]["training"]["image_size"])
file = export(smodel, "severity", torch.zeros(1, 3, size, size), ["logits"])
x = prepare_image(test, size, training=False, rng=np.random.default_rng(0))
with torch.inference_mode():
    expected = smodel(torch.from_numpy(x).unsqueeze(0)).softmax(1)[0].numpy()
got = ort.InferenceSession(str(file)).run(None, {"input": x[None]})[0][0]
got = np.exp(got - got.max()); got /= got.sum()
print(f"severity      softmax max abs diff {np.abs(got - expected).max():.2e}")
assert np.abs(got - expected).max() < 1e-3
manifest["severity"] = {
    "file": "severity.onnx", "sha256": digest, "epoch": sp["epoch"], "size": size,
    "classes": list(SEVERITY_NAMES), "dataset": sp["metadata"]["dataset"],
    "mean": list(MEAN), "std": list(STD),
}

for role in ("segmentation", "crack", "severity"):
    source_file = OUT / manifest[role]["file"]
    short = hashlib.sha256(source_file.read_bytes()).hexdigest()[:10]
    hashed = OUT / f"{role}.{short}.onnx"  # new content => new URL, never a stale cache hit
    source_file.rename(hashed)
    manifest[role]["file"] = hashed.name
    manifest[role]["bytes"] = hashed.stat().st_size
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
for f in sorted(OUT.glob("*.onnx")):
    print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")
