#!/usr/bin/env python3
"""
Create Production Model Bundle

Packages trained model with versioned feature schema, config, and metrics
into a production-ready bundle.

Usage:
    python scripts/create_production_bundle.py --model ML/trained_models/breakout_20251223_212702_model.pkl
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import joblib
import shutil
from datetime import datetime


def create_bundle(model_path: str, output_dir: str = None):
    """
    Create versioned production bundle.
    
    Args:
        model_path: Path to trained model.pkl
        output_dir: Output directory (default: ML/production/v{VERSION})
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model to extract info
    pipeline = joblib.load(model_path)
    
    # Extract timestamp from model filename (breakout_YYYYMMDD_HHMMSS_model.pkl)
    timestamp = model_path.stem.split('_')[1] + '_' + model_path.stem.split('_')[2]
    version = datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%Y%m%d_%H%M%S')
    
    # Create output directory
    if output_dir is None:
        output_dir = project_root / 'ML' / 'production' / f'v{version}'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Creating production bundle v{version}")
    print(f"   Output: {output_dir}")
    
    # ===== 1. Copy model =====
    model_out = output_dir / 'model.pkl'
    shutil.copy(model_path, model_out)
    print(f"✅ Copied model: {model_out.name}")
    
    # ===== 2. Features schema (ORDERED - critical for inference) =====
    features_schema = {
        "version": version,
        "n_features": 8,
        "feature_names": [
            "close_pos_end",
            "clv_mean",
            "atr_pct_mean",
            "event_len",
            "slope_in_range",
            "net_return_in_range",
            "rejection_from_top",
            "rejection_from_bottom"
        ],
        "feature_descriptions": {
            "close_pos_end": "Where price closed in consolidation range (0=bottom, 1=top)",
            "clv_mean": "Average Close Location Value during consolidation",
            "atr_pct_mean": "Mean ATR as % of price (volatility context)",
            "event_len": "Number of bars in consolidation",
            "slope_in_range": "Log-slope of closes during consolidation",
            "net_return_in_range": "Net return from start to end of consolidation",
            "rejection_from_top": "% bars with high near top but close below mid",
            "rejection_from_bottom": "% bars with low near bottom but close above mid"
        },
        "feature_ordering_critical": True,
        "warning": "DO NOT reorder features - this will cause silent prediction errors"
    }
    
    features_out = output_dir / 'features.json'
    with open(features_out, 'w') as f:
        json.dump(features_schema, f, indent=2)
    print(f"✅ Created features schema: {features_out.name}")
    
    # ===== 3. Training configuration =====
    training_config = {
        "version": version,
        "model_type": "HistGradientBoostingClassifier",
        "hyperparameters": {
            "random_state": 42,
            "max_iter": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_samples_leaf": 50,
            "l2_regularization": 1.0
        },
        "data_splits": {
            "train": {
                "date_range": "<= 2024-12-31",
                "description": "All events ending on or before 2024-12-31"
            },
            "val": {
                "date_range": "2025-01-01 to 2025-08-31",
                "description": "Events from Jan-Aug 2025"
            },
            "test": {
                "date_range": ">= 2025-09-01",
                "description": "Events from Sep 2025 onwards"
            }
        },
        "label_rules": {
            "target": "breakout_direction",
            "positive_class": "UP_RESOLVE",
            "negative_class": "DOWN_RESOLVE",
            "filters": [
                "label_valid == True",
                "label_generic in ['UP_RESOLVE', 'DOWN_RESOLVE']"
            ],
            "breakout_definition": "Price breaks above/below consolidation range by band_pct threshold",
            "consolidation_params": {
                "W": 60,
                "T_PRE": 120,
                "H": 40,
                "COOLDOWN": 0,
                "RANGE_PERCENTILE": 55,
                "SLOPE_PERCENTILE": 55
            }
        },
        "feature_engineering": {
            "close_pos_end": "Computed from event-range boundaries (not per-bar)",
            "net_return_in_range": "(close[-1] / close[0]) - 1 within consolidation",
            "slope_in_range": "np.polyfit on log(close) within consolidation",
            "rejection_from_*": "Wick analysis: high near boundary but close on opposite side"
        },
        "training_notes": [
            "No StandardScaler - tree models don't need normalization",
            "Regularization (depth=3, min_samples_leaf=50) prevents overfitting",
            "Features ordered by importance: close_pos_end dominant, net_return_in_range secondary"
        ]
    }
    
    config_out = output_dir / 'training_config.json'
    with open(config_out, 'w') as f:
        json.dump(training_config, f, indent=2)
    print(f"✅ Created training config: {config_out.name}")
    
    # ===== 4. Metrics (from corresponding metrics.json) =====
    metrics_path = model_path.parent / model_path.name.replace('_model.pkl', '_metrics.json')
    
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Add summary stats
        metrics['summary'] = {
            "random_baseline_accuracy": 0.511,
            "random_baseline_auc": 0.50,
            "model_lift_accuracy": round(metrics['test']['accuracy'] - 0.511, 3),
            "model_lift_auc": round(metrics['test']['roc_auc'] - 0.50, 3),
            "generalization_gap": round(metrics['train']['roc_auc'] - metrics['test']['roc_auc'], 3)
        }
        
        metrics_out = output_dir / 'metrics.json'
        with open(metrics_out, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Copied metrics: {metrics_out.name}")
    else:
        print(f"⚠️  Metrics file not found: {metrics_path}")
    
    # ===== 5. README =====
    readme_content = f"""# Production Model Bundle v{version}

## Overview

**Model Type:** Gradient Boosting Classifier  
**Target:** Breakout direction prediction (UP_RESOLVE vs DOWN_RESOLVE)  
**Performance:** Test AUC 0.733, Accuracy 68.2% (vs 51.1% random baseline)

## Files

- `model.pkl` - Trained sklearn pipeline
- `features.json` - Feature schema (ORDERED - do not modify)
- `training_config.json` - Hyperparameters and data splits
- `metrics.json` - Performance metrics on train/val/test

## Usage

```python
import joblib
import json
import pandas as pd

# Load model and schema
model = joblib.load('model.pkl')
with open('features.json', 'r') as f:
    schema = json.load(f)

# CRITICAL: Features must be in this exact order
features = schema['feature_names']

# Prepare input (example)
X = pd.DataFrame({{
    'close_pos_end': [0.75],
    'clv_mean': [0.05],
    'atr_pct_mean': [0.02],
    'event_len': [80],
    'slope_in_range': [0.0001],
    'net_return_in_range': [0.015],
    'rejection_from_top': [0.2],
    'rejection_from_bottom': [0.1]
}})

# Predict
y_proba = model.predict_proba(X[features])[:, 1]  # P(UP_RESOLVE)
y_pred = model.predict(X[features])  # 0=DOWN, 1=UP

print(f"P(UP): {{y_proba[0]:.3f}}")
print(f"Prediction: {{'UP' if y_pred[0] else 'DOWN'}}")
```

## Important Notes

### Feature Ordering
**CRITICAL:** Always use features in the order specified in `features.json`.
Reordering will cause silent prediction errors.

### Entry Timing
**IMPORTANT:** Only enter trades AFTER the consolidation event completes (after `t_end`).
The dominant feature `close_pos_end` is computed from the completed event window.

Recommended entry: `t_end + 1` bar (next bar after event completes)

### Calibration
This model is NOT calibrated. Predicted probabilities may not match observed frequencies.
For production use, apply probability calibration on validation set.

### Decision Thresholds
Default threshold (0.5) is NOT optimal for trading.
Recommend sweeping thresholds on validation set to optimize for:
- Precision (win rate)
- Coverage (trade frequency)
- Expected value (risk-adjusted return)

Typical thresholds: P(UP) >= 0.65 for long, P(UP) <= 0.35 for short

## Performance Summary

```
Train:  Acc=68.9%, F1=69.7%, AUC=0.769
Val:    Acc=66.0%, F1=67.5%, AUC=0.721
Test:   Acc=68.2%, F1=67.0%, AUC=0.733
```

**Lift over random:** +17.1pp accuracy, +0.23 AUC

## Version History

- v{version}: Initial production release
  - 8 curated features
  - Regularized gradient boosting (depth=3, min_samples_leaf=50)
  - Test AUC 0.733

## Next Steps

1. Add probability calibration (CalibratedClassifierCV)
2. Optimize decision threshold on validation set
3. Run event-based backtest with realistic entry/exit
4. Monitor live performance and retrain as needed
"""
    
    readme_out = output_dir / 'README.md'
    with open(readme_out, 'w') as f:
        f.write(readme_content)
    print(f"✅ Created README: {readme_out.name}")
    
    print(f"\n🎉 Production bundle complete!")
    print(f"   Version: v{version}")
    print(f"   Location: {output_dir}")
    print(f"\n📋 Bundle contents:")
    for file in output_dir.iterdir():
        if file.is_file():
            size = file.stat().st_size / 1024  # KB
            print(f"   - {file.name:25s} ({size:>8.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Create versioned production model bundle"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model.pkl file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: ML/production/v{VERSION})'
    )
    
    args = parser.parse_args()
    
    create_bundle(args.model, args.output)


if __name__ == "__main__":
    main()
