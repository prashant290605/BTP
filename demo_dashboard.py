"""Interactive demo dashboard for EWS + drift pipeline.

Run:
    streamlit run demo_dashboard.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data.synthetic_generator import SyntheticTimeSeriesGenerator
from src.evaluation.drift_metrics import compute_all_drift_metrics
from src.evaluation.ews_metrics import (
    compute_false_alarm_rate,
    compute_lead_time,
    compute_regime_separation,
)
from src.models.classical_ews import ClassicalEWS
from src.models.drift_detectors import create_drift_detector


st.set_page_config(page_title="EWS Drift Demo", layout="wide")
plt.style.use("seaborn-v0_8-whitegrid")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.0rem; padding-bottom: 2.0rem;}
    .stMetric {background: rgba(70,130,180,0.12); border-radius: 10px; padding: 0.4rem 0.7rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------- Utility: interpretation helpers ----------------------
def interpret_lead_time(lt: float) -> Tuple[str, str]:
    if not np.isfinite(lt):
        return "No warning before collapse", "🔴"
    if lt < 0:
        return "Alarm after collapse (Bad)", "🔴"
    if lt <= 10:
        return "Very late warning", "🟠"
    if lt <= 40:
        return "Acceptable warning", "🟡"
    return "Strong early warning", "🟢"


def interpret_far(far: float) -> Tuple[str, str]:
    if far > 0.3:
        return "High false alarms", "🔴"
    if far > 0.1:
        return "Moderate false alarms", "🟠"
    return "Low false alarms", "🟢"


def interpret_regime_sep(sep: float) -> Tuple[str, str]:
    if sep < 1:
        return "Weak signal separation", "🔴"
    if sep <= 2:
        return "Moderate separation", "🟡"
    return "Strong signal separation", "🟢"


def interpret_wasserstein(v: float) -> str:
    if v < 0.1:
        return "Negligible drift"
    if v <= 0.5:
        return "Mild drift"
    return "Strong drift"


def interpret_mean_shift(v: float) -> str:
    if v < 0.1:
        return "Negligible shift"
    if v <= 0.5:
        return "Mild shift"
    return "Strong shift"


def interpret_variance_shift(v: float) -> str:
    if v < 0.05:
        return "Negligible shift"
    if v <= 0.25:
        return "Mild shift"
    return "Strong shift"


def interpret_mmd(v: float) -> str:
    if v < 0.05:
        return "Negligible distribution shift"
    if v <= 0.2:
        return "Mild shift"
    return "Strong shift"


def interpret_kl(v: float) -> str:
    if v < 0.2:
        return "Low divergence"
    if v <= 1.0:
        return "Moderate divergence"
    return "High divergence"


# ---------------------- Utility: simulation and models ----------------------
def _derive_regime_lengths(length: int, collapse_time: int) -> Tuple[int, int]:
    """Choose stable_length + transition_length so post-transition starts at collapse_time."""
    collapse_time = int(np.clip(collapse_time, 40, length - 20))
    transition_length = max(30, int(0.2 * length))
    if collapse_time - transition_length < 20:
        transition_length = max(20, collapse_time - 20)
    stable_length = collapse_time - transition_length
    return stable_length, transition_length


def _clamp_to_slider_step(value: int, min_v: int, max_v: int, step: int) -> int:
    """Clamp and align integer value to slider step grid."""
    value = int(np.clip(value, min_v, max_v))
    offset = value - min_v
    aligned = min_v + int(round(offset / step)) * step
    return int(np.clip(aligned, min_v, max_v))


def _ensure_int_slider_state(key: str, min_v: int, max_v: int, default_v: int, step: int) -> None:
    """Prevent streamlit slider value-step conflicts across reruns."""
    if key not in st.session_state:
        st.session_state[key] = _clamp_to_slider_step(default_v, min_v, max_v, step)
    else:
        st.session_state[key] = _clamp_to_slider_step(st.session_state[key], min_v, max_v, step)


def generate_simulation(
    drift_type: str,
    magnitude: float,
    length: int,
    collapse_time: int,
    seed: int,
) -> Dict[str, np.ndarray | int]:
    gen = SyntheticTimeSeriesGenerator(seed=seed)
    stable_length, transition_length = _derive_regime_lengths(length, collapse_time)
    drift_start = int(0.25 * length)
    drift_end = int(0.55 * length)

    if drift_type == "parameter_drift":
        ts, labels = gen.generate_fold_bifurcation(
            length=length,
            stable_length=stable_length,
            transition_length=transition_length,
            control_drift_config={
                "start": drift_start,
                "end": drift_end,
                "magnitude": magnitude,
                "params": {"shift_amount": 0.3, "direction": -1.0, "power": 1.0},
            },
        )
    else:
        ts, labels = gen.generate_fold_bifurcation(
            length=length,
            stable_length=stable_length,
            transition_length=transition_length,
        )
        if drift_type == "mean_shift":
            ts = gen.add_mean_drift(ts, drift_start, drift_end, magnitude=magnitude, shift_amount=0.6)
        elif drift_type == "variance_shift":
            ts = gen.add_variance_drift(
                ts,
                drift_start,
                drift_end,
                magnitude=magnitude,
                initial_scale=1.0,
                final_scale=2.2,
            )

    return {
        "time_series": ts,
        "labels": labels,
        "drift_start": drift_start,
        "drift_end": drift_end,
        "collapse_time": stable_length + transition_length,
    }


def sanitize_detector_params(detector_params: Dict[str, float | int]) -> Dict[str, float | int]:
    int_keys = {"window_size", "min_window_length", "warmup_steps", "clock", "max_window_length"}
    clean: Dict[str, float | int] = {}
    for k, v in detector_params.items():
        clean[k] = int(v) if k in int_keys else float(v)
    return clean


def run_detector(
    signal: np.ndarray,
    detector_name: str,
    detector_params: Dict[str, float | int],
) -> List[int]:
    detector_map = {
        "Variance": "variance",
        "ADWIN": "adwin",
        "CUSUM": "cusum",
        "Page-Hinkley": "page_hinkley",
    }
    detector = create_drift_detector(detector_map[detector_name], **sanitize_detector_params(detector_params))
    alarms: List[int] = []
    for t, v in enumerate(signal):
        if bool(detector.update(float(v))):
            alarms.append(t)
    return alarms


@st.cache_resource(show_spinner=False)
def load_offline_model_cached() -> Tuple[Any, float, float, int]:
    import json
    from tensorflow import keras

    model_path = Path("models/cnn_lstm_offline.keras")
    norm_path = Path("models/cnn_lstm_offline_norm.json")
    if not model_path.exists() or not norm_path.exists():
        raise FileNotFoundError("Offline model files not found in models/.")

    model = keras.models.load_model(str(model_path))
    with open(norm_path, "r", encoding="utf-8") as f:
        norm = json.load(f)
    window_size = int(model.input_shape[1])
    return model, float(norm["mean"]), float(norm["std"]), window_size


@st.cache_resource(show_spinner=False)
def load_online_model_cached(
    detector_type: str,
    detector_params_key: Tuple[Tuple[str, object], ...],
):
    from src.models.online_adaptive_ews import load_online_adaptive_model

    detector_params = {k: v for k, v in detector_params_key}
    model_path = Path("models/cnn_lstm_offline.keras")
    norm_path = Path("models/cnn_lstm_offline_norm.json")
    if not model_path.exists() or not norm_path.exists():
        raise FileNotFoundError("Online model files not found in models/.")
    return load_online_adaptive_model(
        model_path=str(model_path),
        norm_path=str(norm_path),
        detector_type=detector_type,
        detector_params=detector_params,
    )


def compute_offline_scores(signal: np.ndarray) -> np.ndarray:
    model, mean, std, window_size = load_offline_model_cached()
    n = len(signal)
    scores = np.full(n, np.nan, dtype=float)
    if n <= window_size:
        return scores
    windows = np.zeros((n - window_size, window_size, 1), dtype=float)
    for i in range(n - window_size):
        w = (signal[i : i + window_size] - mean) / (std + 1e-12)
        windows[i, :, 0] = w
    preds = model.predict(windows, batch_size=256, verbose=0).flatten()
    scores[window_size:] = preds
    return np.clip(scores, 0.0, 1.0)


def compute_classical_scores(signal: np.ndarray, window_size: int = 40) -> np.ndarray:
    ews = ClassicalEWS(window_size=window_size)
    ts = signal.reshape(1, -1)
    ews.fit(ts)
    scores = ews.predict(signal)
    return np.clip(scores, 0.0, 1.0)


def compute_online_scores(
    signal: np.ndarray,
    labels: np.ndarray,
    detector_name: str,
    detector_params: Dict[str, float | int],
) -> np.ndarray:
    detector_map = {
        "Variance": "variance",
        "ADWIN": "adwin",
        "CUSUM": "cusum",
        "Page-Hinkley": "page_hinkley",
    }
    detector_type = detector_map[detector_name]
    key = tuple(sorted((k, v) for k, v in sanitize_detector_params(detector_params).items()))
    online_model = load_online_model_cached(detector_type, key)
    online_model.reset_stream_state()

    # Fast path using existing model stream processor (batched predictions).
    scores, _ = online_model.process_stream(signal, labels)
    return np.clip(scores, 0.0, 1.0)


def method_scores(
    method_name: str,
    signal: np.ndarray,
    labels: np.ndarray,
    detector_name: str,
    detector_params: Dict[str, float | int],
) -> Tuple[Optional[np.ndarray], str]:
    try:
        if method_name == "Classical EWS":
            return compute_classical_scores(signal), "Computed from Classical EWS."
        if method_name == "Offline CNN-LSTM":
            return compute_offline_scores(signal), "Computed from Offline CNN-LSTM."
        if method_name == "Online Adaptive CNN-LSTM":
            return compute_online_scores(signal, labels, detector_name, detector_params), "Computed from Online Adaptive CNN-LSTM."
        return None, "Unknown method."
    except Exception as exc:  # noqa: BLE001
        return None, f"{method_name} unavailable: {exc}"


def normalize_for_eval(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize one method's score stream for fair threshold comparison."""
    s = scores.copy().astype(float)
    finite = np.isfinite(s)
    if not np.any(finite):
        return s
    smin = np.nanmin(s[finite])
    smax = np.nanmax(s[finite])
    if smax <= smin:
        s[finite] = 0.0
        return s
    s[finite] = (s[finite] - smin) / (smax - smin + 1e-12)
    return s


def deep_model_runtime_status() -> Tuple[bool, str]:
    model_path = Path("models/cnn_lstm_offline.keras")
    norm_path = Path("models/cnn_lstm_offline_norm.json")
    if not model_path.exists() or not norm_path.exists():
        return False, "model files missing in models/."
    try:
        import tensorflow as _tf  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"tensorflow unavailable ({exc})"
    return True, ""


# ---------------------- Sidebar ----------------------
st.title("Interactive EWS + Drift Demonstration Dashboard")
st.caption("Synthetic Drift -> Drift Metrics -> Drift Detection -> Model Warning Scores -> EWS Metrics")

with st.sidebar:
    st.header("Simulation")
    drift_type = st.selectbox("Drift Type", ["mean_shift", "variance_shift", "parameter_drift"])
    magnitude = st.slider("Drift Magnitude", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
    _ensure_int_slider_state("length_slider", 220, 500, 350, 10)
    length = st.slider("Time Series Length", min_value=220, max_value=500, step=10, key="length_slider")
    collapse_max = length - 20
    _ensure_int_slider_state("collapse_slider", 60, collapse_max, int(0.75 * length), 5)
    collapse_time = st.slider("Collapse Time", min_value=60, max_value=collapse_max, step=5, key="collapse_slider")
    seed = st.number_input("Random Seed", min_value=0, max_value=100000, value=42, step=1)

    st.header("Drift Detector")
    detector_name = st.selectbox("Detector", ["Variance", "ADWIN", "CUSUM", "Page-Hinkley"])
    detector_params: Dict[str, float | int] = {}
    if detector_name == "Variance":
        detector_params["threshold"] = st.slider("threshold", 1.1, 5.0, 2.5, 0.1)
        detector_params["window_size"] = st.slider("window_size", 20, 160, 60, 5)
    elif detector_name == "ADWIN":
        detector_params["delta"] = st.slider("delta", 0.0001, 0.1, 0.002, 0.0005)
        detector_params["min_window_length"] = st.slider("min_window_length", 10, 120, 30, 5)
    elif detector_name == "CUSUM":
        detector_params["k"] = st.slider("k", 0.05, 2.0, 0.5, 0.05)
        detector_params["h"] = st.slider("h", 1.0, 30.0, 8.0, 0.5)
        detector_params["warmup_steps"] = st.slider("warmup_steps", 10, 180, 50, 5)
    elif detector_name == "Page-Hinkley":
        detector_params["delta"] = st.slider("delta", 0.0001, 0.1, 0.005, 0.0005)
        detector_params["lambda_"] = st.slider("lambda", 1.0, 50.0, 20.0, 0.5)
        detector_params["warmup_steps"] = st.slider("warmup_steps", 10, 180, 50, 5)

    st.header("Warning Evaluation")
    threshold = st.slider("EWS Threshold", 0.05, 0.95, 0.5, 0.01)
    warning_horizon = st.slider("Warning Horizon", 20, 180, 80, 5)
    pre_collapse_window = st.slider("Pre-collapse Window", 20, 180, 80, 5)

    st.header("Model Comparison")
    selected_method = st.selectbox(
        "Method / Model",
        ["Classical EWS", "Offline CNN-LSTM", "Online Adaptive CNN-LSTM"],
    )
    overlay_methods = st.checkbox("Overlay all methods", value=True)
    normalize_overlay = st.checkbox("Normalize overlay plot (visual only)", value=True)
    eval_mode = st.selectbox(
        "Evaluation Score Mode",
        ["Normalized per method (recommended)", "Raw scores"],
        index=0,
        help="Normalized mode makes cross-method threshold comparison fair in demos.",
    )

    generate_clicked = st.button("Generate Simulation", type="primary")


# ---------------------- Generate/Re-use simulation ----------------------
if generate_clicked or "demo_data" not in st.session_state:
    st.session_state.demo_data = generate_simulation(
        drift_type=drift_type,
        magnitude=magnitude,
        length=length,
        collapse_time=collapse_time,
        seed=int(seed),
    )
    st.session_state.score_cache = {}

data = st.session_state.demo_data
signal = np.asarray(data["time_series"], dtype=float)
labels = np.asarray(data["labels"], dtype=int)
drift_start = int(data["drift_start"])
drift_end = int(data["drift_end"])
collapse_t = int(data["collapse_time"])
t = np.arange(signal.size)


# ---------------------- Panel 1: Synthetic ----------------------
st.subheader("1) Synthetic Drift Generator")
fig1, ax1 = plt.subplots(figsize=(11, 3.5))
ax1.plot(t, signal, color="blue", linewidth=1.3, label="Signal")
ax1.axvline(collapse_t, color="black", linestyle="--", linewidth=1.4, label="Collapse point")
ax1.axvspan(drift_start, drift_end, color="orange", alpha=0.2, label="Drift region")
ax1.set_title(f"Synthetic Signal ({drift_type}, magnitude={magnitude:.2f})")
ax1.set_xlabel("Time")
ax1.set_ylabel("Value")
ax1.legend(loc="upper left")
st.pyplot(fig1, use_container_width=True)
st.caption("Expected behavior: drift region should visibly alter trajectory statistics before collapse.")
st.divider()


# ---------------------- Panel 2: Drift metrics ----------------------
st.subheader("2) Drift Magnitude Metrics")
window = max(20, min(120, signal.size // 4))
baseline = signal[max(0, drift_start - window) : drift_start]
current = signal[drift_end : min(signal.size, drift_end + window)]
if current.size < 10:
    current = signal[-window:]
if baseline.size < 10:
    baseline = signal[:window]

dmetrics = compute_all_drift_metrics(baseline, current)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Wasserstein", f"{dmetrics['wasserstein']:.4f}", help=interpret_wasserstein(dmetrics["wasserstein"]))
c2.metric("KL Divergence", f"{dmetrics['kl_divergence']:.4f}", help=interpret_kl(dmetrics["kl_divergence"]))
c3.metric("MMD", f"{dmetrics['mmd']:.4f}", help=interpret_mmd(dmetrics["mmd"]))
c4.metric("Mean Shift", f"{dmetrics['mean_shift']:.4f}", help=interpret_mean_shift(dmetrics["mean_shift"]))
c5.metric(
    "Variance Shift",
    f"{dmetrics['variance_shift']:.4f}",
    help=interpret_variance_shift(dmetrics["variance_shift"]),
)

st.markdown(
    f"- Wasserstein: **{interpret_wasserstein(dmetrics['wasserstein'])}**\n"
    f"- KL Divergence: **{interpret_kl(dmetrics['kl_divergence'])}**\n"
    f"- MMD: **{interpret_mmd(dmetrics['mmd'])}**\n"
    f"- Mean Shift: **{interpret_mean_shift(dmetrics['mean_shift'])}**\n"
    f"- Variance Shift: **{interpret_variance_shift(dmetrics['variance_shift'])}**"
)

fig2, ax2 = plt.subplots(figsize=(10, 3.2))
names = ["wasserstein", "kl_divergence", "mmd", "mean_shift", "variance_shift"]
vals = [dmetrics[k] for k in names]
ax2.bar(names, vals, color=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"])
ax2.set_title("Drift Metrics: Baseline vs Current Window")
ax2.set_ylabel("Score")
ax2.tick_params(axis="x", rotation=20)
st.pyplot(fig2, use_container_width=True)

# Optional drift metric over time
metric_track_name = st.selectbox(
    "Metric over time",
    ["wasserstein", "mmd", "mean_shift", "variance_shift", "kl_divergence"],
    index=0,
)
track_t, track_vals = [], []
for i in range(drift_start + window, len(signal) + 1):
    cur = signal[i - window : i]
    m = compute_all_drift_metrics(baseline, cur)
    track_t.append(i - 1)
    track_vals.append(m[metric_track_name])
fig2b, ax2b = plt.subplots(figsize=(10, 2.8))
ax2b.plot(track_t, track_vals, color="#4C78A8", linewidth=1.5, label=metric_track_name)
ax2b.axvspan(drift_start, drift_end, color="orange", alpha=0.2, label="Drift region")
ax2b.axvline(collapse_t, color="black", linestyle="--", linewidth=1.2, label="Collapse point")
ax2b.set_title(f"{metric_track_name} over time")
ax2b.set_xlabel("Time")
ax2b.set_ylabel("Metric")
ax2b.legend(loc="upper left")
st.pyplot(fig2b, use_container_width=True)

st.caption("Expected behavior: drift metrics should rise during/after drift region and stay lower in stable region.")
st.divider()


# ---------------------- Panel 3: Drift detection ----------------------
st.subheader("3) Drift Detection Algorithms")
alarms = run_detector(signal, detector_name, detector_params)
first_alarm = alarms[0] if alarms else None
alarms_in_drift = sum(drift_start <= a <= drift_end for a in alarms)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Number of Alarms", len(alarms))
col_b.metric("First Detection Time", "-" if first_alarm is None else str(first_alarm))
col_c.metric("Alarms in Drift Region", alarms_in_drift)

fig3, ax3 = plt.subplots(figsize=(11, 3.5))
ax3.plot(t, signal, color="blue", linewidth=1.2, label="Signal")
if alarms:
    ax3.scatter(np.array(alarms), signal[np.array(alarms)], color="red", marker="^", s=45, label="Drift alarms")
ax3.axvline(collapse_t, color="black", linestyle="--", linewidth=1.4, label="Collapse point")
ax3.axvspan(drift_start, drift_end, color="orange", alpha=0.2, label="Drift region")
ax3.set_title(f"Detector: {detector_name}")
ax3.set_xlabel("Time")
ax3.set_ylabel("Value")
ax3.legend(loc="upper left")
st.pyplot(fig3, use_container_width=True)
st.caption("Expected behavior: alarms should start around the beginning of drift region, not only after collapse.")
st.divider()


# ---------------------- Panel 4: EWS metrics + method comparison ----------------------
st.subheader("4) Early Warning Evaluation Metrics")
deep_ok, deep_msg = deep_model_runtime_status()
if not deep_ok:
    st.info(
        "Offline/Online CNN-LSTM methods are disabled in this runtime: "
        f"{deep_msg} Classical EWS remains fully available."
    )

methods_to_compute = (
    ["Classical EWS", "Offline CNN-LSTM", "Online Adaptive CNN-LSTM"]
    if overlay_methods
    else [selected_method]
)

method_score_map: Dict[str, np.ndarray] = {}
method_msg_map: Dict[str, str] = {}
if "score_cache" not in st.session_state:
    st.session_state.score_cache = {}

cache_prefix = (
    drift_type,
    round(float(magnitude), 4),
    int(length),
    int(collapse_t),
    int(seed),
    detector_name,
    tuple(sorted((k, v) for k, v in sanitize_detector_params(detector_params).items())),
)
for m in methods_to_compute:
    cache_key = (cache_prefix, m)
    if cache_key in st.session_state.score_cache:
        sc, msg = st.session_state.score_cache[cache_key]
    else:
        sc, msg = method_scores(m, signal, labels, detector_name, detector_params)
        st.session_state.score_cache[cache_key] = (sc, msg)
    method_msg_map[m] = msg
    if sc is not None:
        method_score_map[m] = sc

if selected_method not in method_score_map:
    st.warning(method_msg_map.get(selected_method, "Selected method unavailable."))
    selected_scores = np.full_like(signal, np.nan, dtype=float)
else:
    selected_scores_raw = method_score_map[selected_method]
    selected_scores = (
        normalize_for_eval(selected_scores_raw)
        if eval_mode.startswith("Normalized")
        else selected_scores_raw
    )
    st.success(method_msg_map[selected_method])

# Evaluate selected method
lead_time = compute_lead_time(selected_scores, collapse_time=collapse_t, threshold=threshold)
far = compute_false_alarm_rate(
    selected_scores, collapse_time=collapse_t, threshold=threshold, warning_horizon=warning_horizon
)
sep = compute_regime_separation(selected_scores, collapse_time=collapse_t, pre_collapse_window=pre_collapse_window)

lt_text, lt_emoji = interpret_lead_time(lead_time)
far_text, far_emoji = interpret_far(far)
sep_text, sep_emoji = interpret_regime_sep(sep)

m1, m2, m3 = st.columns(3)
m1.metric("Lead Time", "NaN" if not np.isfinite(lead_time) else f"{lead_time:.1f} steps")
m2.metric("False Alarm Rate", f"{far:.4f}")
m3.metric("Regime Separation", f"{sep:.4f}")
st.markdown(
    f"- Lead Time status: {lt_emoji} **{lt_text}**\n"
    f"- False Alarm Rate status: {far_emoji} **{far_text}**\n"
    f"- Regime Separation status: {sep_emoji} **{sep_text}**"
)

# Plot scores
fig4, ax4 = plt.subplots(figsize=(11, 3.8))
if overlay_methods:
    colors = {
        "Classical EWS": "#2ca02c",
        "Offline CNN-LSTM": "#9467bd",
        "Online Adaptive CNN-LSTM": "#1f77b4",
    }
    for name, sc in method_score_map.items():
        y = sc.copy()
        if normalize_overlay:
            finite = np.isfinite(y)
            if np.any(finite):
                ymin = np.nanmin(y)
                ymax = np.nanmax(y)
                if ymax > ymin:
                    y = (y - ymin) / (ymax - ymin + 1e-12)
        ax4.plot(t, y, linewidth=1.6, color=colors.get(name, None), label=name)
else:
    y = selected_scores.copy()
    if normalize_overlay:
        finite = np.isfinite(y)
        if np.any(finite):
            ymin = np.nanmin(y)
            ymax = np.nanmax(y)
            if ymax > ymin:
                y = (y - ymin) / (ymax - ymin + 1e-12)
    ax4.plot(t, y, color="blue", linewidth=1.6, label=selected_method)

ax4.axhline(threshold, color="red", linestyle="-", linewidth=1.2, label=f"Threshold ({threshold:.2f})")
ax4.axvline(collapse_t, color="black", linestyle="--", linewidth=1.4, label="Collapse point")
ax4.axvspan(drift_start, drift_end, color="orange", alpha=0.2, label="Drift region")
ax4.set_title("Warning Score vs Time")
ax4.set_xlabel("Time")
ax4.set_ylabel("Score")
ax4.set_ylim(-0.05, 1.05)
ax4.legend(loc="upper left")
st.pyplot(fig4, use_container_width=True)
if normalize_overlay:
    st.caption("Note: overlay curves are min-max normalized for visual comparison only. Metrics are computed on raw scores.")
st.caption("Expected behavior: score should cross threshold before collapse; lead time should be positive and FAR should stay low.")
st.caption(f"Evaluation mode: **{eval_mode}**")

# Comparison table
rows = []
for name, sc in method_score_map.items():
    sc_eval = normalize_for_eval(sc) if eval_mode.startswith("Normalized") else sc
    lt = compute_lead_time(sc_eval, collapse_time=collapse_t, threshold=threshold)
    fa = compute_false_alarm_rate(
        sc_eval, collapse_time=collapse_t, threshold=threshold, warning_horizon=warning_horizon
    )
    rs = compute_regime_separation(sc_eval, collapse_time=collapse_t, pre_collapse_window=pre_collapse_window)
    lt_s, lt_e = interpret_lead_time(lt)
    fa_s, fa_e = interpret_far(fa)
    rs_s, rs_e = interpret_regime_sep(rs)
    rows.append(
        {
            "Method": name,
            "Lead Time": "NaN" if not np.isfinite(lt) else f"{lt:.1f}",
            "Lead Time Status": f"{lt_e} {lt_s}",
            "False Alarm Rate": f"{fa:.4f}",
            "FAR Status": f"{fa_e} {fa_s}",
            "Regime Separation": f"{rs:.4f}",
            "Separation Status": f"{rs_e} {rs_s}",
            "Raw Score Range": f"[{np.nanmin(sc):.4g}, {np.nanmax(sc):.4g}]",
        }
    )
if rows:
    st.markdown("**Method Comparison Summary**")
    st.table(pd.DataFrame(rows))


# ---------------------- Top-level diagnostic summary ----------------------
st.divider()
st.subheader("System Diagnostic Summary")
drift_detected = dmetrics["wasserstein"] >= 0.1
detector_triggered = len(alarms) > 0
early_warning_generated = bool(
    np.any(np.isfinite(selected_scores[:collapse_t]) & (selected_scores[:collapse_t] >= threshold))
)
collapse_pred_before_event = bool(np.isfinite(lead_time) and lead_time > 0)

s1, s2, s3, s4 = st.columns(4)
s1.metric("Drift Detected", "YES" if drift_detected else "NO")
s2.metric("Detector Triggered", "YES" if detector_triggered else "NO")
s3.metric("Early Warning Generated", "YES" if early_warning_generated else "NO")
s4.metric("Predicted Before Collapse", "YES" if collapse_pred_before_event else "NO")

st.markdown(
    f"- {'✅' if drift_detected else '⚠️'} Drift Detected\n"
    f"- {'✅' if detector_triggered else '⚠️'} Detector Triggered\n"
    f"- {'✅' if early_warning_generated else '⚠️'} Early Warning Generated\n"
    f"- {'✅' if collapse_pred_before_event else '⚠️'} Collapse Predicted Before Event"
)
