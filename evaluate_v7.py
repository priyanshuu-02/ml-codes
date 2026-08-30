"""V7 held-out evaluation, including closed-loop GNSS-outage trajectory drift."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.pytorch_dataset import get_normalizer, load_split
from src.data.targets import prepare_session
from src.data.v7_dataset import build_v7_windows
from src.models.hybrid_v7 import V7DeadReckoningModel


def predict(model, normalizer, imu, initial_speed, device):
    x = normalizer.transform_imu(imu[None]).astype(np.float32)
    state = normalizer.transform_speed(np.asarray([initial_speed], dtype=np.float32)).astype(np.float32)
    with torch.no_grad():
        output = model(torch.from_numpy(x).to(device), torch.from_numpy(state).to(device))
    return (float(normalizer.inverse_speed(output["speed"].cpu().numpy())[0]),
            normalizer.inverse_position(output["position"].cpu().numpy())[0],
            float(normalizer.inverse_yaw(output["yaw_rate"].cpu().numpy())[0]),
            int(output["motion_logits"].argmax(dim=1).item()))


def predict_many(model, normalizer, imu, initial_speed, device, batch_size=512):
    """Batched one-window inference for practical held-out evaluation."""
    x = normalizer.transform_imu(np.asarray(imu, dtype=np.float32)).astype(np.float32)
    state = normalizer.transform_speed(np.asarray(initial_speed, dtype=np.float32)).astype(np.float32)
    speeds, positions, yaws, motions = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = start + batch_size
            output = model(torch.from_numpy(x[start:end]).to(device), torch.from_numpy(state[start:end]).to(device))
            speeds.append(output["speed"].cpu().numpy())
            positions.append(output["position"].cpu().numpy())
            yaws.append(output["yaw_rate"].cpu().numpy())
            motions.append(output["motion_logits"].argmax(dim=1).cpu().numpy())
    return (normalizer.inverse_speed(np.concatenate(speeds)), normalizer.inverse_position(np.concatenate(positions)),
            normalizer.inverse_yaw(np.concatenate(yaws)), np.concatenate(motions))


def closed_loop_metrics(model, normalizer, imu, targets, device, horizon_seconds, window_size=20):
    imu = np.asarray(imu, dtype=np.float32)
    duration = (window_size - 1) / 10.0
    steps = max(1, round(horizon_seconds / duration))
    east = targets["position_east_m"].to_numpy()
    north = targets["position_north_m"].to_numpy()
    heading = targets["heading_deg"].to_numpy()
    speed = targets["speed_mps"].to_numpy()
    anchors = np.arange(0, len(imu) - steps * window_size + 1, window_size, dtype=np.int64)
    if not len(anchors):
        return {"windows": 0, "final_position_error_m": None, "mean_position_error_m": None, "final_heading_error_deg": None}
    pe, pn, ph, ps = east[anchors].copy(), north[anchors].copy(), heading[anchors].copy(), speed[anchors].copy()
    accumulated_error = np.zeros(len(anchors), dtype=np.float64)
    for step in range(steps):
        starts = anchors + step * window_size
        windows = np.stack([imu[start:start + window_size] for start in starts])
        pred_speed, delta, yaw, _ = predict_many(model, normalizer, windows, ps, device)
        theta = np.radians(ph)
        pe += delta[:, 0] * np.sin(theta) + delta[:, 1] * np.cos(theta)
        pn += delta[:, 0] * np.cos(theta) - delta[:, 1] * np.sin(theta)
        ph += np.degrees(yaw * duration); ps = np.maximum(0.0, pred_speed)
        end = starts + window_size - 1
        accumulated_error += np.hypot(pe - east[end], pn - north[end])
    final_error = np.hypot(pe - east[end], pn - north[end])
    heading_error = np.abs((ph - heading[end] + 180) % 360 - 180)
    return {"windows": int(len(anchors)), "final_position_error_m": float(final_error.mean()),
            "mean_position_error_m": float((accumulated_error / steps).mean()), "final_heading_error_deg": float(heading_error.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="v7_state_turn_aware")
    parser.add_argument("--skip-trajectory", action="store_true")
    args = parser.parse_args()
    checkpoint_path = Path("models/checkpoints") / args.experiment / "best_model.pt"
    if not checkpoint_path.exists(): raise FileNotFoundError(f"Missing V7 checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = V7DeadReckoningModel(input_channels=6, conv_dim=96, hidden_dim=128, dropout=0.15).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    normalizer = get_normalizer(); dataset, sessions = load_split("test")
    errors, truth, pred, per_session = [], [], [], {}
    for session in sessions:
        imu_df, target_df = prepare_session(dataset, session)
        raw_imu = imu_df.to_numpy(dtype=np.float32)
        windows = build_v7_windows(raw_imu, target_df)
        sp, pp, yp, mp = predict_many(model, normalizer, windows["imu"], windows["initial_speed"], device)
        position_error = pp - windows["position"]
        per_session[session["session_id"]] = {"samples": len(pp), "forward_mae_m": float(np.abs(position_error[:, 0]).mean()),
            "lateral_mae_m": float(np.abs(position_error[:, 1]).mean()), "motion_accuracy": float((mp == windows["motion"]).mean())}
        errors.append(position_error); truth.append(windows); pred.append((sp, yp, mp))
    position_error = np.concatenate(errors)
    all_windows = {key: np.concatenate([w[key] for w in truth]) for key in ("speed", "yaw_rate", "motion")}
    all_pred_speed = np.concatenate([p[0] for p in pred]); all_pred_yaw = np.concatenate([p[1] for p in pred]); all_pred_motion = np.concatenate([p[2] for p in pred])
    result = {"checkpoint": str(checkpoint_path), "test_set_used_for_selection": False,
              "speed_mae_mps": float(np.abs(all_pred_speed - all_windows["speed"]).mean()),
              "forward_mae_m": float(np.abs(position_error[:, 0]).mean()), "lateral_mae_m": float(np.abs(position_error[:, 1]).mean()),
              "position_rmse_m": float(np.sqrt(np.mean(position_error ** 2))), "yaw_mae_rad_s": float(np.abs(all_pred_yaw - all_windows["yaw_rate"]).mean()),
              "motion_accuracy": float((all_pred_motion == all_windows["motion"]).mean()), "per_session": per_session, "trajectory": {}}
    if not args.skip_trajectory:
        for horizon in (10, 20, 30, 60):
            by_session = {s["session_id"]: closed_loop_metrics(model, normalizer, *prepare_session(dataset, s), device, horizon) for s in sessions}
            valid = [x["final_position_error_m"] for x in by_session.values() if x["final_position_error_m"] is not None]
            result["trajectory"][f"{horizon}s"] = {"mean_final_position_error_m": float(np.mean(valid)) if valid else None, "per_session": by_session}
    out = Path("outputs") / args.experiment / "test_results.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
