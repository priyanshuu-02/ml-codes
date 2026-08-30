"""Export a validated V7 checkpoint for mobile/runtime integration."""
import argparse
import json
import shutil
from pathlib import Path

import torch

from src.models.hybrid_v7 import V7DeadReckoningModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="v7_state_turn_aware_run2")
    parser.add_argument("--output", default="models/deploy/v7")
    args = parser.parse_args()
    checkpoint_path = Path("models/checkpoints") / args.experiment / "best_model.pt"
    result_path = Path("outputs") / args.experiment / "test_results.json"
    if not checkpoint_path.exists(): raise FileNotFoundError(checkpoint_path)
    if not result_path.exists(): raise RuntimeError("Run evaluate_v7.py and review its test results before export.")
    results = json.loads(result_path.read_text(encoding="utf-8"))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    model = V7DeadReckoningModel(input_channels=6, conv_dim=96, hidden_dim=128, dropout=0.15)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]); model.eval()
    sample_imu, sample_speed = torch.zeros(1, 20, 6), torch.zeros(1)
    torch.onnx.export(model, (sample_imu, sample_speed), out / "model.onnx", input_names=["imu", "initial_speed_normalized"],
                      output_names=["speed", "speed_log_variance", "position", "position_log_variance", "yaw_rate", "yaw_rate_log_variance", "motion_logits"],
                      dynamic_axes={"imu": {0: "batch"}, "initial_speed_normalized": {0: "batch"}}, opset_version=17)
    shutil.copy2("models/preprocessing/normalization.json", out / "normalization.json")
    (out / "deployment_manifest.json").write_text(json.dumps({"architecture": "V7 state-conditioned", "imu_shape": [20, 6],
        "sample_rate_hz": 10, "window_seconds": 2, "initial_speed": "last trusted GNSS/DR state, normalized with normalization.json",
        "selection": "validation only", "held_out_test_results": results}, indent=2), encoding="utf-8")
    print(f"Exported deployment candidate to {out}")


if __name__ == "__main__": main()
