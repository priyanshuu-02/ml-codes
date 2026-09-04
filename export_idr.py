"""
Export a trained IDR-V1 checkpoint to ONNX for the Android runtime.

Output names are fixed and must match what V8DeadReckoningEngine looks up, so the runtime
reads the new model with no change beyond adding the acceleration head. Input names are
unchanged for the same reason.

The exported graph is verified against PyTorch before the artifact is written, because a
model that loads successfully is not the same thing as a model that computes the same
values.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.idr_dataset import (
    CHANNEL_NAMES,
    SAMPLE_RATE_HZ,
    STRIDE_SAMPLES,
    VERSION,
    WINDOW_SAMPLES,
    WINDOW_SPAN_SECONDS,
)
from src.models.idr_v1 import IdrV1Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="idr_v1_full")
    parser.add_argument("--output", default="models/deploy/idr_v1")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    checkpoint_path = Path("models/checkpoints") / args.experiment / "best_model.pt"
    normalization_path = Path("outputs") / args.experiment / "normalization.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not normalization_path.exists():
        raise FileNotFoundError(normalization_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = IdrV1Model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "idr_v1.onnx"

    sample_imu = torch.zeros(1, WINDOW_SAMPLES, len(CHANNEL_NAMES))
    sample_speed = torch.zeros(1)

    torch.onnx.export(
        model,
        (sample_imu, sample_speed),
        onnx_path,
        input_names=IdrV1Model.input_names(),
        output_names=IdrV1Model.output_names(),
        dynamic_axes={"imu": {0: "batch"}, "initial_speed_normalized": {0: "batch"}},
        opset_version=args.opset,
        # The dynamo exporter in torch 2.10 fails on the GRU's implicit zero hidden state:
        # aten.unbind receives a symbolic tensor it cannot iterate. The legacy TorchScript
        # exporter handles recurrent layers correctly, and its output is what ONNX Runtime
        # 1.18 on the device consumes.
        dynamo=False,
    )
    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KiB)")

    # Verify the graph reproduces PyTorch on random input rather than trusting the export.
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    generator = torch.Generator().manual_seed(0)
    test_imu = torch.randn(8, WINDOW_SAMPLES, len(CHANNEL_NAMES), generator=generator)
    test_speed = torch.randn(8, generator=generator)

    with torch.no_grad():
        expected = model(test_imu, test_speed)

    actual = session.run(
        IdrV1Model.output_names(),
        {"imu": test_imu.numpy(), "initial_speed_normalized": test_speed.numpy()},
    )

    print("\nparity check against PyTorch")
    worst = 0.0
    for name, torch_value, onnx_value in zip(IdrV1Model.output_names(), expected, actual):
        difference = float(np.max(np.abs(torch_value.numpy() - onnx_value)))
        worst = max(worst, difference)
        print(f"  {name:<28} max abs diff {difference:.3e}")
    if worst > 1e-4:
        raise RuntimeError(f"ONNX output diverges from PyTorch by {worst:.3e}")
    print(f"  worst {worst:.3e}  PASS")

    shutil.copy2(normalization_path, output_dir / "normalization.json")

    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    manifest = {
        "model": "MARK-V IDR-V1 motion model",
        "preprocessing_version": VERSION,
        "input": {
            "imu": [WINDOW_SAMPLES, len(CHANNEL_NAMES)],
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "stride_samples": STRIDE_SAMPLES,
            "window_span_seconds": WINDOW_SPAN_SECONDS,
            "channels": CHANNEL_NAMES,
            "frame": "PHONE_LINEAR",
            "gravity": "REMOVED",
            "gyro_order": "DATASET_NATIVE",
            "initial_speed": "last trusted GNSS or DR speed, normalised with normalization.json",
        },
        "outputs": IdrV1Model.output_names(),
        "parameters": model.parameter_count(),
        "sha256": digest,
        "selection": "validation only; the test session was never read during training",
        "trained_epoch": checkpoint.get("epoch"),
        "validation_metrics": checkpoint.get("validation_metrics"),
        "training_sessions": ["S3c", "S1", "S2", "M"],
        "validation_sessions": ["S3a"],
        "test_sessions": ["Y1"],
        "known_limitations": [
            "Trained on 4 sessions from the 6 of 72 whose smartphone IMU demonstrably "
            "corresponds to the vehicle log; the remaining 66 are not usable.",
            "Phone-frame input, so it is not mount independent. A vehicle-frame model is "
            "not trainable from IO-VNBD because its gyro columns behave as Euler angle "
            "rates rather than a rotatable vector.",
            "One held-out test session, so generalisation evidence is limited to a single "
            "unseen driver and vehicle.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nparameters : {model.parameter_count():,}")
    print(f"sha256     : {digest}")
    print(f"manifest   : {output_dir / 'manifest.json'}")
    print(f"normaliser : {output_dir / 'normalization.json'}")


if __name__ == "__main__":
    main()
