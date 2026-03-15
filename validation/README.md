# validation/

This folder contains one-time diagnostic scripts that verify major project
components behave as expected.

## Scripts

| Script | What it validates |
|---|---|
| `validate_drift_generators.py` | Mean drift, variance drift, and control-parameter drift produce expected statistical changes. |
| `validate_drift_metrics.py` | Wasserstein, KL, MMD, mean-shift, and variance-shift metrics are near zero for identical distributions and scale with drift. |
| `validate_drift_detectors.py` | ADWIN, CUSUM, Page-Hinkley, and Variance detectors detect a known shift within tolerance. Pre-shift alarms are reported (not required to be zero). |
| `validate_ews_metrics.py` | Lead time behavior, FAR monotonicity, and regime-separation monotonicity. |

## Run (from project root)

```bash
python validation/validate_drift_generators.py
python validation/validate_drift_metrics.py
python validation/validate_drift_detectors.py
python validation/validate_ews_metrics.py
```

## Outputs

All outputs are written to `validation/outputs/`:

- `drift_generators.png`
- `drift_metrics.png`
- `drift_detectors.png`
- `ews_metrics.png`
- `drift_generators.log`
- `drift_metrics.log`
- `drift_detectors.log`
- `ews_metrics.log`

These scripts are diagnostic and separate from the main experiment pipeline.

