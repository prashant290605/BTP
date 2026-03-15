# -*- coding: utf-8 -*-
"""
validate_drift_detectors.py
===========================
One-time validation: confirm that ADWIN, CUSUM, Page-Hinkley, and Variance
drift detectors fire near a known shift point in a synthetic stream.

Run from the project root:
    python validation/validate_drift_detectors.py

What is checked
---------------
For each detector:
  1. At least one detection event occurs after the shift point.
  2. First detection happens within N_TOLERANCE steps after the shift.
     (Loose tolerance: within 40% of the post-shift window length.)
  3. Number and rate of false alarms (detections BEFORE the shift) are reported.

A timeline plot is saved to  validation/outputs/drift_detectors.png
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.drift_detectors import (
    ADWINDetector,
    CUSUMDetector,
    PageHinkleyDetector,
    VarianceDriftDetector,
)

RNG = np.random.default_rng(0)

# Build a synthetic stream with one clear mean shift
STABLE_LEN = 300     # stable region: N(0, 1)
POST_LEN   = 400     # post-shift region: N(3, 1) -- 3-sigma jump
SHIFT_PT   = STABLE_LEN

stable_stream  = RNG.normal(0.0, 1.0, STABLE_LEN)
shifted_stream = RNG.normal(3.0, 1.0, POST_LEN)
stream = np.concatenate([stable_stream, shifted_stream])
N = len(stream)

N_TOLERANCE = int(0.4 * POST_LEN)   # accept detection within 40% of post window

detectors = {
    "ADWIN":        ADWINDetector(delta=0.002, min_window_length=30),
    "CUSUM":        CUSUMDetector(k=0.5, h=5.0, warmup_steps=50),
    "Page-Hinkley": PageHinkleyDetector(delta=0.005, lambda_=10.0, warmup_steps=50),
    "Variance":     VarianceDriftDetector(threshold=2.0, window_size=60),
}

detection_flags = {name: [] for name in detectors}
first_detection = {}

for name, det in detectors.items():
    for val in stream:
        fired = det.update(val)
        detection_flags[name].append(int(fired))

    flags = np.array(detection_flags[name])
    post_detections = np.where(flags[SHIFT_PT:] == 1)[0]
    pre_detections  = np.where(flags[:SHIFT_PT] == 1)[0]

    first_post = int(post_detections[0]) if post_detections.size > 0 else None
    first_detection[name] = first_post

    detected   = first_post is not None
    timely     = detected and (first_post <= N_TOLERANCE)
    first_str  = ("T+" + str(first_post)) if detected else "NONE"
    result_str = "PASS" if timely else ("LATE" if detected else "NOT_DETECTED")

    pre_far = pre_detections.size / SHIFT_PT
    print(
        f"{name:<16s} | detected={str(detected):<5s}  "
        f"first_detect_step={first_str:>8s}  "
        f"timely (<={N_TOLERANCE})={result_str}  "
        f"pre_shift_alarms={pre_detections.size}  pre_shift_far={pre_far:.3f}"
    )

# Plot
os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), "outputs", "drift_detectors.png")

n_det = len(detectors)
fig, axes = plt.subplots(n_det + 1, 1, figsize=(14, 3 + 2 * n_det), sharex=True)
fig.suptitle(
    "Drift Detector Validation  (Shift at t=300, stream N(0,1) -> N(3,1))",
    fontsize=13, fontweight="bold"
)

t = np.arange(N)

axes[0].plot(t, stream, linewidth=0.6, color="steelblue")
axes[0].axvline(SHIFT_PT, color="red", linewidth=1.5, linestyle="--", label="True shift")
axes[0].set_title("Synthetic stream")
axes[0].legend(fontsize=8)
axes[0].set_ylabel("x(t)")

for i, name in enumerate(detectors):
    ax = axes[i + 1]
    flags = np.array(detection_flags[name])
    ax.fill_between(t, 0, flags, alpha=0.6, step="mid", label="Drift alarm")
    ax.axvline(SHIFT_PT, color="red", linewidth=1.5, linestyle="--", alpha=0.5)
    fd = first_detection[name]
    if fd is not None:
        ax.axvline(SHIFT_PT + fd, color="green", linewidth=1.5,
                   linestyle=":", label=f"1st detect T+{fd}")
    ax.set_title(name)
    ax.set_ylim(-0.1, 1.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["no", "alarm"])
    ax.legend(fontsize=8, loc="upper left")

axes[-1].set_xlabel("Time step")
plt.tight_layout()
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nPlot saved -> {out_path}")
