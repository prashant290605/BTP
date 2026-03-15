# -*- coding: utf-8 -*-
"""
validate_drift_metrics.py
=========================
One-time validation: confirm that drift magnitude metrics behave correctly.

Run from the project root:
    python validation/validate_drift_metrics.py

What is checked
---------------
1. Zero drift  -- all metrics == 0 when baseline is compared to itself.
2. Monotonicity -- distribution-sensitive metrics increase as mean shift grows.
   (variance_shift is NOT expected to correlate with mean shift -- it is checked
   against variance drift instead.)
3. Variance increase -- variance-sensitive metrics increase with std scale factor.

Plots saved to  validation/outputs/drift_metrics.png
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.drift_metrics import compute_all_drift_metrics

RNG = np.random.default_rng(42)
N   = 1000
BASE_MU, BASE_STD = 0.0, 1.0

# ---- 1. Zero drift: compare array with itself ------------------------------
baseline = RNG.normal(BASE_MU, BASE_STD, N)
m0 = compute_all_drift_metrics(baseline, baseline.copy())

tol = 1e-6
zero_checks = {k: v < tol for k, v in m0.items()}
print("-- Zero drift check (baseline vs itself, exact zeros expected) --")
all_zero_pass = True
for name, val in m0.items():
    ok = zero_checks[name]
    if not ok: all_zero_pass = False
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<22s} = {val:.2e}  (expected < {tol:.0e})")

# ---- 2. Monotonicity with MEAN shift ---------------------------------------
# Metrics that should increase: wasserstein, kl_divergence, mmd, mean_shift
# variance_shift should stay near 0 (mean-only shift doesn't change variance)
magnitudes = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
mean_sensitive = ["wasserstein", "kl_divergence", "mmd", "mean_shift"]
all_metric_names = mean_sensitive + ["variance_shift"]
results = {n: [] for n in all_metric_names}

for delta in magnitudes:
    cur = RNG.normal(BASE_MU + delta, BASE_STD, N)
    m   = compute_all_drift_metrics(baseline, cur)
    for n in all_metric_names:
        results[n].append(m[n])

print("\n-- Monotonicity with mean shift (expected r > 0.80 for mean-sensitive metrics) --")
monotone_checks = {}
all_mono_pass = True
for n in mean_sensitive:
    vals = np.array(results[n])
    r = float(np.corrcoef(magnitudes, vals)[0, 1])
    monotone_checks[n] = r
    ok = r > 0.80
    if not ok: all_mono_pass = False
    print(f"  {'PASS' if ok else 'FAIL'}  {n:<22s} r = {r:.4f}  (expected > 0.80)")

# variance_shift should have low correlation with mean-only shift
vs_vals = np.array(results["variance_shift"])
vs_r = float(np.corrcoef(magnitudes, vs_vals)[0, 1]) if np.std(vs_vals) > 1e-10 else 0.0
vs_ok = abs(vs_r) < 0.5
print(f"  {'PASS' if vs_ok else 'FAIL'}  {'variance_shift':<22s} r = {vs_r:.4f}  (expected near 0 -- measures variance, not mean)")

# ---- 3. Monotonicity with VARIANCE drift -----------------------------------
# All metrics except mean_shift should increase with variance
std_factors = [1.0, 1.5, 2.0, 3.0, 4.0]
var_sensitive = ["variance_shift", "wasserstein", "kl_divergence", "mmd"]
var_results = {n: [] for n in all_metric_names}

for sf in std_factors:
    cur = RNG.normal(BASE_MU, BASE_STD * sf, N)
    m   = compute_all_drift_metrics(baseline, cur)
    for n in all_metric_names:
        var_results[n].append(m[n])

print("\n-- Variance drift (std scale factor, expected r > 0.70 for variance-sensitive metrics) --")
all_var_pass = True
for n in var_sensitive:
    vals = var_results[n]
    r = float(np.corrcoef(std_factors, vals)[0, 1])
    ok = r > 0.70
    if not ok: all_var_pass = False
    print(f"  {'PASS' if ok else 'FAIL'}  {n:<22s} r = {r:.4f}  (expected > 0.70)")

# ---- Plot ------------------------------------------------------------------
os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), "outputs", "drift_metrics.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Drift Metrics Validation", fontsize=13, fontweight="bold")

ax = axes[0]
for n in all_metric_names:
    ax.plot(magnitudes, results[n], marker="o", label=n)
ax.set_xlabel("Mean shift magnitude (delta)")
ax.set_ylabel("Metric value")
ax.set_title("Metrics vs. Mean Shift")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
for n in all_metric_names:
    ax.plot(std_factors, var_results[n], marker="s", label=n)
ax.set_xlabel("Std scale factor")
ax.set_ylabel("Metric value")
ax.set_title("Metrics vs. Variance Drift")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nPlot saved -> {out_path}")

print(f"\nZero-drift:    {'All PASSED' if all_zero_pass else 'Some FAILED'}")
print(f"Mean monotone: {'All PASSED' if all_mono_pass else 'Some FAILED'}")
print(f"Var monotone:  {'All PASSED' if all_var_pass else 'Some FAILED'}")
overall = all_zero_pass and all_mono_pass and all_var_pass
print(f"\nOverall: {'ALL CHECKS PASSED' if overall else 'SOME CHECKS FAILED'}")
