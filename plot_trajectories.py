"""
Figures for the Stage 8 blackout ablation.

Everything plotted here is read from outputs/blackout_ablation.json, which is written by the
Kotlin test BlackoutAblationTest while it drives the real estimator. Nothing is recomputed,
so a figure cannot quietly disagree with the table it illustrates.

Produces:
  outputs/blackout_error_vs_time.png    median error and drift against elapsed outage time
  outputs/blackout_trajectory_*.png     ground truth against each configuration, in metres
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("outputs")
ABLATION = OUTPUT_DIR / "blackout_ablation.json"
PREDICTIONS = Path("../app/src/test/resources/blackout/predictions.json")

EARTH_RADIUS_METERS = 6371000.0

# Only the configurations worth putting on a chart. The full table is in the test output.
TRAJECTORY_SERIES = [
    ("truth", "ground truth", "#111111", 2.4, "-"),
    ("A", "V8 + EKF", "#c0392b", 1.4, "--"),
    ("P", "persistence", "#7f8c8d", 1.2, ":"),
    ("B", "IDR-V1 + EKF", "#2980b9", 1.6, "-"),
    ("Dn", "IDR-V1 + NHC + map", "#16a085", 1.8, "-"),
    ("G1", "IDR-V1 + gyro heading", "#8e44ad", 1.3, "-."),
]

# Width descends through the list so that where two configurations coincide the earlier,
# thicker one stays visible underneath. B and C coincide almost exactly, and that overlap is
# the result rather than a plotting artefact: the non-holonomic constraint changes nothing
# measurable on this session.
CURVE_SERIES = [
    ("P", "persistence", "#7f8c8d", ":", 1.6),
    ("A", "V8 + EKF", "#c0392b", "--", 1.6),
    ("B", "IDR-V1 + EKF", "#2980b9", "-", 3.4),
    ("C", "+ non-holonomic (coincides with B)", "#f39c12", "-", 1.3),
    ("Dn", "+ map constraint", "#16a085", "-", 2.0),
    ("G1", "gyro heading (production default)", "#8e44ad", "-.", 1.6),
]


def to_local(latitudes, longitudes, origin_lat, origin_lon):
    scale_north = EARTH_RADIUS_METERS * np.pi / 180.0
    scale_east = scale_north * np.cos(np.radians(origin_lat))
    north = (np.asarray(latitudes) - origin_lat) * scale_north
    east = (np.asarray(longitudes) - origin_lon) * scale_east
    return east, north


def plot_error_curves(data):
    curves = data["errorCurves"]
    figure, (error_axis, drift_axis) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    crossings = {}
    for key, label, colour, style, width in CURVE_SERIES:
        curve = curves.get(key)
        if not curve:
            continue
        elapsed = [point["elapsedSeconds"] for point in curve]
        drift = [point["medianDriftPercent"] for point in curve]
        error_axis.plot(
            elapsed, [point["medianErrorMeters"] for point in curve],
            label=label, color=colour, linestyle=style, linewidth=width
        )
        drift_axis.plot(elapsed, drift, label=label, color=colour, linestyle=style, linewidth=width)
        crossings[key] = crossing_time(elapsed, drift, 10.0)

    blackouts = data["blackouts"]
    error_axis.set_xlabel("elapsed outage time (s)")
    error_axis.set_ylabel("median position error (m)")
    error_axis.set_title(f"Position error during GNSS outage\nsession {data['session']}, "
                         f"{blackouts} blackouts, median across blackouts")
    error_axis.grid(alpha=0.3)
    error_axis.legend(fontsize=8)

    drift_axis.axhline(10.0, color="#c0392b", linewidth=1.4, linestyle="--")
    drift_axis.text(
        0.02, 11.0, "SIH target 10 %", color="#c0392b", fontsize=9,
        ha="left", va="bottom", transform=drift_axis.get_yaxis_transform()
    )
    best = crossings.get("Dn")
    drift_axis.set_xlabel("elapsed outage time (s)")
    drift_axis.set_ylabel("median drift (% of distance travelled)")
    subtitle = ("best configuration stays under the target for only "
                f"{best:.0f} s" if best else "no configuration reaches the target")
    drift_axis.set_title(f"Drift as a fraction of distance travelled\n{subtitle}")
    drift_axis.grid(alpha=0.3)
    drift_axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.92))

    figure.tight_layout()
    path = OUTPUT_DIR / "blackout_error_vs_time.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")

    print()
    print("outage duration for which median drift stays under 10 percent")
    for key, label, _, _, _ in CURVE_SERIES:
        if key not in crossings:
            continue
        value = crossings[key]
        text = "never under the target" if value is None else f"{value:.1f} s"
        print(f"  {key:<3} {label:<38} {text}")


def crossing_time(elapsed, values, threshold):
    """
    Last moment the curve is still below `threshold`, linearly interpolated.

    Returns None when the curve starts above the threshold and never gets below it.
    """
    if not values or values[0] >= threshold:
        return None
    for index in range(1, len(values)):
        if values[index] >= threshold:
            span = values[index] - values[index - 1]
            if span <= 0:
                return elapsed[index - 1]
            fraction = (threshold - values[index - 1]) / span
            return elapsed[index - 1] + fraction * (elapsed[index] - elapsed[index - 1])
    return elapsed[-1]


def plot_trajectory(data, trajectory, index, road_ways):
    origin_lat = trajectory["seedLat"]
    origin_lon = trajectory["seedLon"]
    estimates = trajectory["estimates"]

    figure, axis = plt.subplots(figsize=(8.6, 8.0))

    truth = np.asarray(trajectory["truth"], dtype=float)
    truth_east, truth_north = to_local(truth[:, 0], truth[:, 1], origin_lat, origin_lon)

    # Road proxy, clipped to the area actually driven. Drawn faintly and labelled as a proxy
    # so it is never mistaken for real OSM geometry.
    if road_ways:
        margin = 200.0
        east_low, east_high = truth_east.min() - margin, truth_east.max() + margin
        north_low, north_high = truth_north.min() - margin, truth_north.max() + margin
        drawn = False
        for way in road_ways:
            points = np.asarray(way["points"], dtype=float)
            east, north = to_local(points[:, 0], points[:, 1], origin_lat, origin_lon)
            if east.max() < east_low or east.min() > east_high:
                continue
            if north.max() < north_low or north.min() > north_high:
                continue
            axis.plot(
                east, north, color="#bdc3c7", linewidth=4.0, solid_capstyle="round",
                zorder=1, label="road proxy (optimistic)" if not drawn else None
            )
            drawn = True

    for key, label, colour, width, style in TRAJECTORY_SERIES:
        if key == "truth":
            east, north = truth_east, truth_north
        else:
            series = estimates.get(key)
            if not series:
                continue
            points = np.asarray(series, dtype=float)
            east, north = to_local(points[:, 0], points[:, 1], origin_lat, origin_lon)
        axis.plot(east, north, color=colour, linewidth=width, linestyle=style, label=label, zorder=3)

    axis.plot(0.0, 0.0, marker="o", markersize=9, color="#27ae60", zorder=5,
              label="outage begins (last GNSS fix)")
    axis.plot(truth_east[-1], truth_north[-1], marker="*", markersize=15, color="#111111",
              zorder=5, label="true end position")

    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("east of last fix (m)")
    axis.set_ylabel("north of last fix (m)")
    duration = len(truth) * data["spanSeconds"]
    axis.set_title(
        f"Open-loop trajectory through a {duration:.0f} s GNSS outage\n"
        f"session {data['session']} ({trajectory['label']}), "
        f"reference final error {trajectory['finalErrorMeters']:.0f} m"
    )
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="best")

    figure.tight_layout()
    path = OUTPUT_DIR / f"blackout_trajectory_{index}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")


def main():
    if not ABLATION.exists():
        print(f"{ABLATION} not found. Run the Kotlin ablation first:")
        print("  cd .. && .\\gradlew :app:testDebugUnitTest "
              "--tests \"nisargpatel.deadreckoning.BlackoutAblationTest\"")
        return 1

    data = json.loads(ABLATION.read_text(encoding="utf-8"))

    road_ways = []
    if PREDICTIONS.exists():
        road_ways = json.loads(PREDICTIONS.read_text(encoding="utf-8"))["road"]["ways"]
    else:
        print(f"note: {PREDICTIONS} not found, trajectories will omit the road proxy")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_error_curves(data)
    for index, trajectory in enumerate(data["trajectories"], start=1):
        plot_trajectory(data, trajectory, index, road_ways)

    print()
    print("reference configuration:", data["referenceConfiguration"])
    print("limitations carried from the export:")
    for number, limitation in enumerate(data["limitations"], start=1):
        print(f"  {number}. {limitation}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
