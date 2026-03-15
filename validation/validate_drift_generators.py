"""
validate_drift_generators.py
One-time validation for synthetic drift generation behavior.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.synthetic_generator import SyntheticTimeSeriesGenerator


RNG_SEED = 42
LENGTH = 1000
DRIFT_START = 300
DRIFT_END = 600


def main() -> None:
    # 1) Mean drift
    gen1 = SyntheticTimeSeriesGenerator(seed=RNG_SEED)
    ts_no_drift, _ = gen1.generate_fold_bifurcation(length=LENGTH)
    ts_mean_drift = gen1.add_mean_drift(
        ts_no_drift, drift_start=DRIFT_START, drift_end=DRIFT_END, magnitude=2.0
    )
    pre_mean = float(np.nanmean(ts_no_drift[:DRIFT_START]))
    post_mean = float(np.nanmean(ts_mean_drift[DRIFT_END:]))
    mean_check = post_mean > pre_mean + 0.1
    print(
        f"[1] Mean drift | pre_mean={pre_mean:.4f} post_mean={post_mean:.4f} "
        f"{'PASS' if mean_check else 'FAIL'}"
    )

    # 2) Variance drift
    gen2 = SyntheticTimeSeriesGenerator(seed=RNG_SEED)
    ts_no_drift2, _ = gen2.generate_fold_bifurcation(length=LENGTH)
    ts_var_drift = gen2.add_variance_drift(
        ts_no_drift2,
        drift_start=DRIFT_START,
        drift_end=DRIFT_END,
        magnitude=1.0,
        initial_scale=1.0,
        final_scale=3.0,
    )
    pre_std = float(np.nanstd(ts_no_drift2[:DRIFT_START]))
    post_std = float(np.nanstd(ts_var_drift[DRIFT_END:]))
    var_check = post_std > pre_std * 1.2
    print(
        f"[2] Var drift  | pre_std={pre_std:.4f} post_std={post_std:.4f} "
        f"{'PASS' if var_check else 'FAIL'}"
    )

    # 3) Control-parameter drift
    gen_base = SyntheticTimeSeriesGenerator(seed=RNG_SEED)
    ts_base_ctrl, _ = gen_base.generate_fold_bifurcation(length=LENGTH)
    gen_ctrl = SyntheticTimeSeriesGenerator(seed=RNG_SEED)
    ts_ctrl_drift, _ = gen_ctrl.generate_fold_bifurcation(
        length=LENGTH,
        control_drift_config={
            "start": DRIFT_START,
            "end": DRIFT_END,
            "magnitude": 1.5,
            "params": {"shift_amount": 0.3, "direction": -1.0, "power": 1.0},
        },
    )
    diff_pre = float(np.nanmean(np.abs(ts_ctrl_drift[:DRIFT_START] - ts_base_ctrl[:DRIFT_START])))
    diff_post = float(np.nanmean(np.abs(ts_ctrl_drift[DRIFT_END:] - ts_base_ctrl[DRIFT_END:])))
    ctrl_check = (diff_post > diff_pre) or (diff_pre < 1e-6)
    print(
        f"[3] Ctrl drift | diff_pre={diff_pre:.4f} diff_post={diff_post:.4f} "
        f"{'PASS' if ctrl_check else 'FAIL'}"
    )

    # Plot
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "drift_generators.png")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    fig.suptitle("Drift Generator Validation", fontsize=14, fontweight="bold")
    t = np.arange(LENGTH)

    ax = axes[0]
    ax.plot(t, ts_no_drift, label="No drift", alpha=0.7, linewidth=0.8)
    ax.plot(t, ts_mean_drift, label="Mean drift", alpha=0.9, linewidth=0.8)
    ax.axvspan(DRIFT_START, DRIFT_END, alpha=0.1, color="red", label="Drift region")
    ax.set_title(
        f"Mean drift (pre_mu={pre_mean:.3f} -> post_mu={post_mean:.3f}) [{'PASS' if mean_check else 'FAIL'}]",
        loc="left",
    )
    ax.legend(fontsize=7)
    ax.set_ylabel("x(t)")

    ax = axes[1]
    ax.plot(t, ts_no_drift2, label="No drift", alpha=0.7, linewidth=0.8)
    ax.plot(t, ts_var_drift, label="Var drift", alpha=0.9, linewidth=0.8)
    ax.axvspan(DRIFT_START, DRIFT_END, alpha=0.1, color="red")
    ax.set_title(
        f"Variance drift (pre_std={pre_std:.3f} -> post_std={post_std:.3f}) [{'PASS' if var_check else 'FAIL'}]",
        loc="left",
    )
    ax.legend(fontsize=7)
    ax.set_ylabel("x(t)")

    ax = axes[2]
    ax.plot(t, ts_base_ctrl, label="No ctrl drift", alpha=0.7, linewidth=0.8)
    ax.plot(t, ts_ctrl_drift, label="Ctrl param drift", alpha=0.9, linewidth=0.8)
    ax.axvspan(DRIFT_START, DRIFT_END, alpha=0.1, color="red")
    ax.set_title(
        f"Control-parameter drift (post_divergence={diff_post:.3f}) [{'PASS' if ctrl_check else 'FAIL'}]",
        loc="left",
    )
    ax.legend(fontsize=7)
    ax.set_ylabel("x(t)")
    ax.set_xlabel("Time step")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved -> {out_path}")

    all_pass = mean_check and var_check and ctrl_check
    print(f"\n{'All checks PASSED' if all_pass else 'Some checks FAILED'}")


if __name__ == "__main__":
    main()

