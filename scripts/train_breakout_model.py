#!/usr/bin/env python3
"""
Train Breakout Direction Model

Trains a gradient boosting classifier to predict breakout direction (up vs down)
from consolidation patterns in hourly OHLCV data.

Usage:
    python scripts/train_breakout_model.py --input data/ml_datasets/accum_distrib_events.parquet
    python scripts/train_breakout_model.py --input data/ml_datasets/accum_distrib_events.parquet --target accumulation
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix,
    classification_report
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BreakoutModelTrainer:
    """Train and evaluate breakout direction prediction model."""
    
    # Curated feature set - removed harmful features, kept helpful ones
    FEATURE_COLS = [
        # Primary signal
        'close_pos_end',
        
        # Secondary helpful features
        'clv_mean',
        'atr_pct_mean',
        
        # Event structure (hard to fake)
        'event_len',
        'slope_in_range',
        'net_return_in_range',
        
        # Wick rejection (pressure signals)
        'rejection_from_top',
        'rejection_from_bottom'
    ]
    
    def __init__(self, target_mode: str = 'breakout'):
        """
        Initialize trainer.
        
        Args:
            target_mode: 'breakout' for UP/DOWN resolution, 'accumulation' for ACCUM/DIST
        """
        self.target_mode = target_mode
        self.pipeline = None
        self.metrics = {}
        
    def load_and_prepare_data(self, input_path: str) -> tuple:
        """
        Load dataset and prepare train/val/test splits.
        
        Args:
            input_path: Path to parquet file
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("=" * 80)
        logger.info(f"🚀 Training Breakout Direction Model - {self.target_mode.upper()} target")
        logger.info("=" * 80)
        logger.info(f"📥 Loading dataset from {input_path}")
        
        df = pd.read_parquet(input_path)
        
        logger.info(f"✅ Loaded {len(df):,} events")
        logger.info(f"   Date range: {df['t_end'].min()} to {df['t_end'].max()}")
        logger.info(f"   Tickers: {df['ticker'].nunique()}")
        
        # Filter based on target mode
        if self.target_mode == 'breakout':
            # Option A: Breakout direction (binary)
            df_filtered = df[
                (df['label_valid'] == True) & 
                (df['label_generic'].isin(['UP_RESOLVE', 'DOWN_RESOLVE']))
            ].copy()
            
            df_filtered['target'] = (df_filtered['label_generic'] == 'UP_RESOLVE').astype(int)
            target_names = ['DOWN_RESOLVE', 'UP_RESOLVE']
            logger.info("\n📊 Target: Breakout Direction (binary)")
            logger.info("   0 = DOWN_RESOLVE, 1 = UP_RESOLVE")
            
        elif self.target_mode == 'accumulation':
            # Option B: Accumulation vs Distribution
            df_filtered = df[
                (df['label_valid'] == True) & 
                (df['label_acc_dist'].isin(['ACCUMULATION', 'DISTRIBUTION']))
            ].copy()
            
            df_filtered['target'] = (df_filtered['label_acc_dist'] == 'ACCUMULATION').astype(int)
            target_names = ['DISTRIBUTION', 'ACCUMULATION']
            logger.info("\n📊 Target: Accumulation/Distribution (binary)")
            logger.info("   0 = DISTRIBUTION, 1 = ACCUMULATION")
        else:
            raise ValueError(f"Invalid target_mode: {self.target_mode}")
        
        logger.info(f"✅ Filtered to {len(df_filtered):,} valid events ({len(df_filtered)/len(df)*100:.1f}%)")
        logger.info(f"   Class balance: {df_filtered['target'].value_counts().to_dict()}")
        
        # Time-based split
        df_filtered['t_end_date'] = pd.to_datetime(df_filtered['t_end']).dt.date
        
        train_cutoff = pd.to_datetime('2024-12-31').date()
        val_cutoff = pd.to_datetime('2025-08-31').date()
        test_start = pd.to_datetime('2025-09-01').date()
        
        train = df_filtered[df_filtered['t_end_date'] <= train_cutoff].copy()
        val = df_filtered[
            (df_filtered['t_end_date'] > train_cutoff) &
            (df_filtered['t_end_date'] <= val_cutoff)
        ].copy()
        test = df_filtered[df_filtered['t_end_date'] >= test_start].copy()
        
        logger.info(f"\n📅 Time-based splits (no leakage):")
        logger.info(f"   Train: <= 2024-12-31  ({len(train):,} events, {len(train)/len(df_filtered)*100:.1f}%)")
        logger.info(f"   Val:   2025-01-01 to 2025-08-31  ({len(val):,} events, {len(val)/len(df_filtered)*100:.1f}%)")
        logger.info(f"   Test:  >= 2025-09-01  ({len(test):,} events, {len(test)/len(df_filtered)*100:.1f}%)")
        
        # Check class balance per split
        for split_name, split_df in [('Train', train), ('Val', val), ('Test', test)]:
            if len(split_df) > 0:
                balance = split_df['target'].value_counts(normalize=True).to_dict()
                logger.info(f"   {split_name} balance: {balance}")
        
        self.target_names = target_names
        
        return train, val, test
    
    def train_model(self, train: pd.DataFrame, val: pd.DataFrame):
        """
        Train gradient boosting model with validation monitoring.
        
        Args:
            train: Training DataFrame
            val: Validation DataFrame
        """
        logger.info("\n" + "=" * 80)
        logger.info("🎯 Training Model")
        logger.info("=" * 80)
        
        X_train = train[self.FEATURE_COLS]
        y_train = train['target']
        X_val = val[self.FEATURE_COLS]
        y_val = val['target']
        
        logger.info(f"Features: {len(self.FEATURE_COLS)} numeric features")
        logger.info(f"Training samples: {len(X_train):,}")
        logger.info(f"Validation samples: {len(X_val):,}")
        
        # Check for missing values
        train_nulls = X_train.isnull().sum().sum()
        val_nulls = X_val.isnull().sum().sum()
        if train_nulls > 0 or val_nulls > 0:
            logger.warning(f"⚠️  Found {train_nulls} train nulls, {val_nulls} val nulls - filling with 0")
            X_train = X_train.fillna(0)
            X_val = X_val.fillna(0)
        
        # Build pipeline with regularization
        self.pipeline = Pipeline([
            ('model', HistGradientBoostingClassifier(
                random_state=42,
                max_iter=200,
                learning_rate=0.05,
                max_depth=3,              # Simpler trees
                min_samples_leaf=50,      # More regularization
                l2_regularization=1.0,    # L2 penalty
                verbose=0
            ))
        ])
        
        logger.info("\n🔧 Model: HistGradientBoostingClassifier (REGULARIZED)")
        logger.info("   - max_iter=200, learning_rate=0.05")
        logger.info("   - max_depth=3 (simpler), min_samples_leaf=50, l2_regularization=1.0")
        logger.info("   - No scaler (tree models don't need normalization)")
        
        logger.info("\n⏳ Training... (this may take a minute)")
        self.pipeline.fit(X_train, y_train)
        
        logger.info(f"✅ Training complete!")
        
    def evaluate_model(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
        """
        Evaluate model on all splits and return metrics.
        
        Args:
            train: Training DataFrame
            val: Validation DataFrame
            test: Test DataFrame
            
        Returns:
            Dict of metrics
        """
        logger.info("\n" + "=" * 80)
        logger.info("📊 Evaluation Results")
        logger.info("=" * 80)
        
        metrics = {
            'target_mode': self.target_mode,
            'n_features': len(self.FEATURE_COLS),
            'feature_names': self.FEATURE_COLS,
            'target_names': self.target_names
        }
        
        for split_name, split_df in [('train', train), ('val', val), ('test', test)]:
            if len(split_df) == 0:
                logger.warning(f"⚠️  Skipping {split_name} - no data")
                continue
            
            X = split_df[self.FEATURE_COLS].fillna(0)
            y_true = split_df['target']
            
            y_pred = self.pipeline.predict(X)
            y_proba = self.pipeline.predict_proba(X)[:, 1]
            
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_proba)
            cm = confusion_matrix(y_true, y_pred)
            
            metrics[split_name] = {
                'accuracy': float(acc),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
                'confusion_matrix': cm.tolist(),
                'n_samples': int(len(y_true)),
                'class_balance': y_true.value_counts().to_dict()
            }
            
            logger.info(f"\n{split_name.upper()} SET:")
            logger.info(f"  Samples: {len(y_true):,}")
            logger.info(f"  Accuracy: {acc:.4f}")
            logger.info(f"  F1 Score: {f1:.4f}")
            logger.info(f"  ROC-AUC:  {roc_auc:.4f}")
            logger.info(f"  Confusion Matrix:")
            logger.info(f"    {cm[0].tolist()}  <- {self.target_names[0]}")
            logger.info(f"    {cm[1].tolist()}  <- {self.target_names[1]}")
            logger.info(f"           ↑          ↑")
            logger.info(f"      Pred {self.target_names[0][:4]}  Pred {self.target_names[1][:4]}")
        
        # Feature importance (top 10)
        model = self.pipeline.named_steps['model']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = sorted(
                zip(self.FEATURE_COLS, importances),
                key=lambda x: x[1],
                reverse=True
            )
            
            logger.info("\n🔝 Top 10 Feature Importances:")
            for i, (feat, imp) in enumerate(feature_importance[:10], 1):
                logger.info(f"  {i:2d}. {feat:25s} {imp:.4f}")
            
            metrics['feature_importance'] = [
                {'feature': feat, 'importance': float(imp)}
                for feat, imp in feature_importance
            ]
        
        self.metrics = metrics
        return metrics
    
    def save_artifacts(self, output_dir: str):
        """
        Save trained model, metrics, and metadata.
        
        Args:
            output_dir: Directory to save artifacts
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = f"{self.target_mode}_{timestamp}"
        
        # Save model pipeline
        model_file = output_path / f"{prefix}_model.pkl"
        joblib.dump(self.pipeline, model_file)
        logger.info(f"\n💾 Saved model to {model_file}")
        
        # Save metrics
        metrics_file = output_path / f"{prefix}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"💾 Saved metrics to {metrics_file}")
        
        # Save feature list
        features_file = output_path / f"{prefix}_features.txt"
        with open(features_file, 'w') as f:
            f.write('\n'.join(self.FEATURE_COLS))
        logger.info(f"💾 Saved feature list to {features_file}")
        
        # Save training summary
        summary_file = output_path / f"{prefix}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Breakout Direction Model - {self.target_mode.upper()}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Trained: {datetime.now().isoformat()}\n")
            f.write(f"Target: {self.target_mode}\n")
            f.write(f"Features: {len(self.FEATURE_COLS)}\n\n")
            
            for split in ['train', 'val', 'test']:
                if split in self.metrics:
                    m = self.metrics[split]
                    f.write(f"{split.upper()} SET:\n")
                    f.write(f"  Samples: {m['n_samples']:,}\n")
                    f.write(f"  Accuracy: {m['accuracy']:.4f}\n")
                    f.write(f"  F1 Score: {m['f1_score']:.4f}\n")
                    f.write(f"  ROC-AUC:  {m['roc_auc']:.4f}\n\n")
        
        logger.info(f"💾 Saved summary to {summary_file}")
        logger.info(f"\n✅ All artifacts saved to {output_path}/")


def main():
    parser = argparse.ArgumentParser(
        description="Train breakout direction prediction model"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/ml_datasets/accum_distrib_events.parquet',
        help='Input parquet file with events'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        choices=['breakout', 'accumulation'],
        default='breakout',
        help='Target mode: breakout (UP/DOWN) or accumulation (ACCUM/DIST)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='ML/trained_models',
        help='Output directory for artifacts'
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = BreakoutModelTrainer(target_mode=args.target)
    
    # Load and prepare data
    train, val, test = trainer.load_and_prepare_data(args.input)
    
    if len(train) == 0:
        logger.error("❌ No training data available!")
        sys.exit(1)
    
    # Train model
    trainer.train_model(train, val)
    
    # Evaluate
    trainer.evaluate_model(train, val, test)
    
    # Save artifacts
    trainer.save_artifacts(args.output)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Training pipeline complete!")
    logger.info("=" * 80)
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review metrics in {args.output}/")
    logger.info(f"  2. Check feature importance to understand model behavior")
    logger.info(f"  3. If validation performance is good, use model for predictions")
    logger.info(f"  4. Consider hyperparameter tuning if needed")


if __name__ == "__main__":
    main()
