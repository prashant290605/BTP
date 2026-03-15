"""
validate_ews_metrics.py
One-time validation for lead time, false alarm rate, and regime separation.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.ews_metrics import (
    compute_false_alarm_rate,
    compute_lead_time,
    compute_regime_separation,
)


RNG = np.random.default_rng(7)
COLLAPSE_TIME = 800
T = 1000
THRESHOLD = 0.5


def make_scores(
    stable_val: float, precollapse_val: float, noise: float = 0.05, ramp_start: int = 600
) -> np.ndarray:
    """Create synthetic score trajectory with a pre-collapse ramp."""
    s = np.full(T, stable_val, dtype=float)
    ramp = np.linspace(stable_val, precollapse_val, COLLAPSE_TIME - ramp_start)
    s[ramp_start:COLLAPSE_TIME] = ramp
    s += RNG.normal(0.0, noise, T)
    return np.clip(s, 0.0, 1.0)


def main() -> None:
    # 1) Lead time should be positive when warning appears before collapse.
    scores_good = make_scores(stable_val=0.1, precollapse_val=0.9, ramp_start=550)
    lt_good = compute_lead_time(scores_good, collapse_time=COLLAPSE_TIME, threshold=THRESHOLD)
    lt_check = np.isfinite(lt_good) and lt_good > 0
    print(
        f"[1] Lead time (good signal) = {lt_good:.1f} "
        f"{'PASS positive and finite' if lt_check else 'FAIL'}"
    )

    # 2) Lead time should be NaN when there is no alarm.
    scores_flat = np.full(T, 0.1)
    lt_none = compute_lead_time(scores_flat, collapse_time=COLLAPSE_TIME, threshold=THRESHOLD)
    lt_none_check = np.isnan(lt_none)
    print(
        f"[2] Lead time (no signal) = {lt_none} "
        f"{'PASS NaN as expected' if lt_none_check else 'FAIL expected NaN'}"
    )

    # 3) FAR should increase as early alarm fraction increases.
    far_vals = []
    fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    for frac in fractions:
        n_alarm = int(frac * (COLLAPSE_TIME - 200))
        s = np.full(T, 0.1)
        idx = RNG.choice(COLLAPSE_TIME - 200, size=n_alarm, replace=False)
        s[idx] = 0.9
        far = compute_false_alarm_rate(
            s, collapse_time=COLLAPSE_TIME, threshold=THRESHOLD, warning_horizon=200
        )
        far_vals.append(far)

    far_mono = all(far_vals[i] <= far_vals[i + 1] + 1e-6 for i in range(len(far_vals) - 1))
    print(f"\n[3] FAR monotone with alarm fraction: {'PASS' if far_mono else 'FAIL'}")
    for frac, far in zip(fractions, far_vals):
        print(f"    frac={frac:.2f} FAR={far:.4f}")

    # 4) Regime separation should increase with stronger pre-collapse scores.
    pre_means = [0.1, 0.2, 0.4, 0.6, 0.8]
    sep_vals = []
    for pm in pre_means:
        s = make_scores(stable_val=0.1, precollapse_val=pm, noise=0.02, ramp_start=600)
        sep = compute_regime_separation(s, collapse_time=COLLAPSE_TIME, pre_collapse_window=200)
        sep_vals.append(sep)

    sep_mono = all(sep_vals[i] <= sep_vals[i + 1] + 1e-6 for i in range(len(sep_vals) - 1))
    print(f"\n[4] Regime separation monotone with pre-collapse mean: {'PASS' if sep_mono else 'FAIL'}")
    for pm, sep in zip(pre_means, sep_vals):
        print(f"    pre_mean={pm:.1f} separation={sep:.4f}")

    # Plot
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ews_metrics.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("EWS Metrics Validation", fontsize=13, fontweight="bold")
    t = np.arange(T)

    ax = axes[0, 0]
    ax.plot(t, scores_good, linewidth=0.8, label="Score")
    ax.axhline(THRESHOLD, color="orange", linestyle="--", label=f"Threshold {THRESHOLD}")
    ax.axvline(COLLAPSE_TIME, color="red", linewidth=1.5, linestyle="--", label="Collapse")
    if np.isfinite(lt_good):
        ax.annotate(
            f"Lead time = {lt_good:.0f}",
            xy=(COLLAPSE_TIME - lt_good, THRESHOLD),
            xytext=(COLLAPSE_TIME - lt_good - 100, THRESHOLD + 0.15),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=8,
        )
    ax.set_title(f"Lead time ({'PASS' if lt_check else 'FAIL'})")
    ax.legend(fontsize=8)
    ax.set_ylabel("Score")
    ax.set_xlabel("Time")

    ax = axes[0, 1]
    ax.plot(t, scores_flat, linewidth=0.8, color="gray", label="Score (flat)")
    ax.axhline(THRESHOLD, color="orange", linestyle="--", label=f"Threshold {THRESHOLD}")
    ax.axvline(COLLAPSE_TIME, color="red", linewidth=1.5, linestyle="--", label="Collapse")
    ax.set_title(f"No-alarm lead time NaN ({'PASS' if lt_none_check else 'FAIL'})")
    ax.legend(fontsize=8)
    ax.set_ylabel("Score")
    ax.set_xlabel("Time")

    ax = axes[1, 0]
    ax.plot(fractions, far_vals, marker="o", color="steelblue")
    ax.set_xlabel("Fraction of stable region in alarm")
    ax.set_ylabel("False Alarm Rate")
    ax.set_title(f"FAR monotonicity ({'PASS' if far_mono else 'FAIL'})")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(pre_means, sep_vals, marker="s", color="green")
    ax.set_xlabel("Pre-collapse mean score")
    ax.set_ylabel("Regime separation")
    ax.set_title(f"Separation monotonicity ({'PASS' if sep_mono else 'FAIL'})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved -> {out_path}")

    all_pass = lt_check and lt_none_check and far_mono and sep_mono
    print(f"\n{'All checks PASSED' if all_pass else 'Some checks FAILED'}")


if __name__ == "__main__":
    main()

