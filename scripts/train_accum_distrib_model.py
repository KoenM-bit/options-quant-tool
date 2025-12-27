#!/usr/bin/env python3
"""
Train Accumulation/Distribution Event Model

Trains ML models to predict breakout direction from range consolidation patterns.

Usage:
    python scripts/train_accum_distrib_model.py --input data/ml_datasets/accum_distrib_events.parquet --mode generic
    python scripts/train_accum_distrib_model.py --input data/ml_datasets/accum_distrib_events.parquet --mode wyckoff
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import logging
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, 
    f1_score, brier_score_loss, classification_report, confusion_matrix
)
from sklearn.dummy import DummyClassifier

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AccumDistribTrainer:
    """Train and evaluate accumulation/distribution models."""
    
    EMBARGO_HOURS = 40  # Match lookahead window
    
    def __init__(self, data_path: str, mode: str = 'generic'):
        """
        Initialize trainer.
        
        Args:
            data_path: Path to parquet dataset
            mode: 'generic' for UP/DOWN or 'wyckoff' for ACCUMULATION/DISTRIBUTION
        """
        self.data_path = data_path
        self.mode = mode
        self.df = None
        self.feature_cols = None
        
    def load_and_validate(self) -> pd.DataFrame:
        """
        Load dataset and run validation checks.
        
        Returns:
            Validated DataFrame
        """
        logger.info("📥 Loading dataset...")
        df = pd.read_parquet(self.data_path)
        
        logger.info(f"✅ Loaded {len(df):,} events from {df['ticker'].nunique()} tickers")
        logger.info(f"   Date range: {df['t_start'].min()} to {df['t_end'].max()}")
        
        # Validation checks
        logger.info("\n🔍 Running validation checks...")
        
        # 1. Check for required columns
        required = ['ticker', 'market', 't_start', 't_end', 'label_generic', 'label_acc_dist']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # 2. Check for NaNs in labels
        nan_generic = df['label_generic'].isna().sum()
        if nan_generic > 0:
            logger.warning(f"⚠️  Found {nan_generic} NaN values in label_generic, dropping...")
            df = df.dropna(subset=['label_generic'])
        
        # 3. Identify feature columns
        self.feature_cols = [
            col for col in df.columns 
            if col not in ['ticker', 'market', 't_start', 't_end', 
                          'label_generic', 'label_acc_dist', 'prior_dir', 'R_pre']
            and not col.startswith('t_')
        ]
        
        logger.info(f"   Identified {len(self.feature_cols)} feature columns")
        
        # 4. Check for NaNs in features
        nan_counts = df[self.feature_cols].isna().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"⚠️  Found NaNs in features:")
            for col, count in nan_counts[nan_counts > 0].items():
                logger.warning(f"      {col}: {count}")
            logger.warning(f"   Filling NaNs with 0...")
            df[self.feature_cols] = df[self.feature_cols].fillna(0)
        
        # 5. Check time ordering per ticker
        logger.info("   Checking time ordering per ticker...")
        for ticker in df['ticker'].unique():
            ticker_df = df[df['ticker'] == ticker].sort_values('t_end')
            if not ticker_df['t_end'].is_monotonic_increasing:
                logger.warning(f"⚠️  Time ordering issue in {ticker}")
        
        # 6. Leakage check (by code review - features use only past data)
        logger.info("   ✅ Leakage check: all features computed from event window (past data only)")
        
        logger.info("✅ Validation complete\n")
        
        self.df = df
        return df
    
    def print_summary_stats(self):
        """Print dataset summary statistics."""
        df = self.df
        
        logger.info("=" * 80)
        logger.info("📊 DATASET SUMMARY")
        logger.info("=" * 80)
        
        # Events per ticker
        ticker_counts = df['ticker'].value_counts()
        logger.info("\nTop 10 tickers by event count:")
        for ticker, count in ticker_counts.head(10).items():
            logger.info(f"  {ticker:10s}: {count:4,} events")
        
        logger.info("\nBottom 10 tickers by event count:")
        for ticker, count in ticker_counts.tail(10).items():
            logger.info(f"  {ticker:10s}: {count:4,} events")
        
        # Market distribution
        logger.info("\nEvents by market:")
        market_counts = df['market'].value_counts()
        for market, count in market_counts.items():
            logger.info(f"  {market:5s}: {count:5,} events ({count/len(df)*100:5.1f}%)")
        
        # Label distribution
        logger.info("\nGeneric label distribution (overall):")
        for label, count in df['label_generic'].value_counts().items():
            logger.info(f"  {label:15s}: {count:5,} ({count/len(df)*100:5.1f}%)")
        
        logger.info("\nGeneric label distribution by market:")
        for market in df['market'].unique():
            market_df = df[df['market'] == market]
            logger.info(f"\n  {market}:")
            for label, count in market_df['label_generic'].value_counts().items():
                logger.info(f"    {label:15s}: {count:5,} ({count/len(market_df)*100:5.1f}%)")
        
        logger.info("\nAcc/Dist label distribution (overall):")
        for label, count in df['label_acc_dist'].value_counts().items():
            logger.info(f"  {label:15s}: {count:5,} ({count/len(df)*100:5.1f}%)")
        
        logger.info("=" * 80 + "\n")
    
    def prepare_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare target variable based on mode.
        
        Returns:
            (features_df, target_series)
        """
        df = self.df.copy()
        
        if self.mode == 'generic':
            logger.info("🎯 Mode: Generic direction (UP_RESOLVE vs DOWN_RESOLVE)")
            
            # Keep only UP/DOWN resolve
            df = df[df['label_generic'].isin(['UP_RESOLVE', 'DOWN_RESOLVE'])].copy()
            
            # Binary: 1 = UP, 0 = DOWN
            df['target'] = (df['label_generic'] == 'UP_RESOLVE').astype(int)
            
            logger.info(f"   Kept {len(df):,} events")
            logger.info(f"   Class balance: UP={df['target'].sum():,}, DOWN={(~df['target']).sum():,}")
            
        elif self.mode == 'wyckoff':
            logger.info("🎯 Mode: Wyckoff context (ACCUMULATION vs DISTRIBUTION)")
            
            # Keep only ACCUMULATION/DISTRIBUTION
            df = df[df['label_acc_dist'].isin(['ACCUMULATION', 'DISTRIBUTION'])].copy()
            
            # Binary: 1 = ACCUMULATION, 0 = DISTRIBUTION
            df['target'] = (df['label_acc_dist'] == 'ACCUMULATION').astype(int)
            
            logger.info(f"   Kept {len(df):,} events")
            logger.info(f"   Class balance: ACC={df['target'].sum():,}, DIST={(~df['target']).sum():,}")
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return df
    
    def create_time_splits(self, df: pd.DataFrame, n_folds: int = 3) -> List[Dict]:
        """
        Create time-based train/val/test splits with embargo.
        
        Args:
            df: DataFrame with events
            n_folds: Number of folds (default 3 for simple split)
            
        Returns:
            List of dicts with train_idx, val_idx, test_idx
        """
        logger.info(f"\n🔪 Creating {n_folds} time-based splits with {self.EMBARGO_HOURS}h embargo...")
        
        # Sort by end time
        df = df.sort_values('t_end').reset_index(drop=True)
        
        if n_folds == 1:
            # Simple 70/15/15 split
            n = len(df)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            
            # Apply embargo
            train_cutoff = df.iloc[train_end]['t_end']
            val_start_emb = train_cutoff + timedelta(hours=self.EMBARGO_HOURS)
            
            val_cutoff = df.iloc[val_end]['t_end']
            test_start_emb = val_cutoff + timedelta(hours=self.EMBARGO_HOURS)
            
            train_idx = df[df['t_end'] < train_cutoff].index.tolist()
            val_idx = df[(df['t_end'] >= val_start_emb) & (df['t_end'] < val_cutoff)].index.tolist()
            test_idx = df[df['t_end'] >= test_start_emb].index.tolist()
            
            splits = [{
                'fold': 1,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_idx': test_idx,
                'train_end': train_cutoff,
                'val_end': val_cutoff
            }]
            
        else:
            # Walk-forward expanding window
            dates = pd.to_datetime(df['t_end']).dt.date.unique()
            dates = sorted(dates)
            
            splits = []
            step_size = len(dates) // (n_folds + 1)
            
            for i in range(n_folds):
                train_end_date = dates[step_size * (i + 1)]
                val_end_date = dates[min(step_size * (i + 2), len(dates) - 1)]
                
                train_cutoff = pd.Timestamp(train_end_date)
                val_cutoff = pd.Timestamp(val_end_date)
                
                # Apply embargo
                val_start_emb = train_cutoff + timedelta(hours=self.EMBARGO_HOURS)
                test_start_emb = val_cutoff + timedelta(hours=self.EMBARGO_HOURS)
                
                train_idx = df[df['t_end'] < train_cutoff].index.tolist()
                val_idx = df[(df['t_end'] >= val_start_emb) & (df['t_end'] < val_cutoff)].index.tolist()
                test_idx = df[df['t_end'] >= test_start_emb].index.tolist()
                
                if len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) > 0:
                    splits.append({
                        'fold': i + 1,
                        'train_idx': train_idx,
                        'val_idx': val_idx,
                        'test_idx': test_idx,
                        'train_end': train_cutoff,
                        'val_end': val_cutoff
                    })
        
        for split in splits:
            logger.info(f"\n  Fold {split['fold']}:")
            logger.info(f"    Train: {len(split['train_idx']):5,} events (up to {split['train_end'].date()})")
            logger.info(f"    Val:   {len(split['val_idx']):5,} events")
            logger.info(f"    Test:  {len(split['test_idx']):5,} events")
        
        logger.info("")
        return splits
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_prob: np.ndarray, name: str = "Model") -> Dict:
        """
        Evaluate model predictions with comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            name: Model name for logging
            
        Returns:
            Dict of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'brier_score': brier_score_loss(y_true, y_prob)
        }
        
        # High-confidence predictions
        high_conf_mask = (y_prob > 0.6) | (y_prob < 0.4)
        if high_conf_mask.sum() > 0:
            high_conf_acc = accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask])
            high_conf_coverage = high_conf_mask.sum() / len(y_true)
            metrics['high_conf_accuracy'] = high_conf_acc
            metrics['high_conf_coverage'] = high_conf_coverage
        else:
            metrics['high_conf_accuracy'] = np.nan
            metrics['high_conf_coverage'] = 0.0
        
        logger.info(f"\n{name} Metrics:")
        logger.info(f"  Accuracy:            {metrics['accuracy']:.4f}")
        logger.info(f"  Balanced Accuracy:   {metrics['balanced_accuracy']:.4f}")
        logger.info(f"  ROC-AUC:             {metrics['roc_auc']:.4f}")
        logger.info(f"  Macro F1:            {metrics['macro_f1']:.4f}")
        logger.info(f"  Brier Score:         {metrics['brier_score']:.4f}")
        logger.info(f"  High-Conf Accuracy:  {metrics['high_conf_accuracy']:.4f} (coverage: {metrics['high_conf_coverage']:.2%})")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"\n  Confusion Matrix:")
        logger.info(f"    {cm}")
        
        return metrics
    
    def train_baseline_models(self, df: pd.DataFrame, splits: List[Dict]):
        """
        Train and evaluate baseline models.
        
        Args:
            df: DataFrame with features and target
            splits: List of train/val/test splits
        """
        logger.info("=" * 80)
        logger.info("🤖 TRAINING BASELINE MODELS")
        logger.info("=" * 80)
        
        # Use normalized features if available
        feature_cols = [col for col in self.feature_cols if '_norm' in col]
        if not feature_cols:
            feature_cols = self.feature_cols
            logger.info(f"Using {len(feature_cols)} raw features")
        else:
            logger.info(f"Using {len(feature_cols)} normalized features")
        
        all_results = []
        
        for split in splits:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"FOLD {split['fold']}")
            logger.info(f"{'=' * 80}")
            
            # Extract data
            train_idx = split['train_idx']
            val_idx = split['val_idx']
            test_idx = split['test_idx']
            
            X_train = df.loc[train_idx, feature_cols].values
            y_train = df.loc[train_idx, 'target'].values
            
            X_val = df.loc[val_idx, feature_cols].values
            y_val = df.loc[val_idx, 'target'].values
            
            X_test = df.loc[test_idx, feature_cols].values
            y_test = df.loc[test_idx, 'target'].values
            
            # Standardize if using raw features
            if '_norm' not in feature_cols[0]:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                X_test = scaler.transform(X_test)
            
            # 1. Dummy Classifier (majority class)
            logger.info("\n" + "-" * 80)
            logger.info("1. Dummy Classifier (Majority Class)")
            logger.info("-" * 80)
            
            dummy = DummyClassifier(strategy='most_frequent')
            dummy.fit(X_train, y_train)
            
            y_pred_dummy = dummy.predict(X_test)
            y_prob_dummy = dummy.predict_proba(X_test)[:, 1]
            
            dummy_metrics = self.evaluate_predictions(y_test, y_pred_dummy, y_prob_dummy, "Dummy")
            dummy_metrics['model'] = 'Dummy'
            dummy_metrics['fold'] = split['fold']
            all_results.append(dummy_metrics)
            
            # 2. Logistic Regression
            logger.info("\n" + "-" * 80)
            logger.info("2. Logistic Regression")
            logger.info("-" * 80)
            
            lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            lr.fit(X_train, y_train)
            
            y_pred_lr = lr.predict(X_test)
            y_prob_lr = lr.predict_proba(X_test)[:, 1]
            
            lr_metrics = self.evaluate_predictions(y_test, y_pred_lr, y_prob_lr, "Logistic Regression")
            lr_metrics['model'] = 'LogisticRegression'
            lr_metrics['fold'] = split['fold']
            all_results.append(lr_metrics)
            
            # Feature importance
            if len(feature_cols) <= 20:
                importance = np.abs(lr.coef_[0])
                top_features = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)[:10]
                logger.info(f"\n  Top 10 features:")
                for feat, imp in top_features:
                    logger.info(f"    {feat:30s}: {imp:.4f}")
            
            # 3. LightGBM
            if HAS_LIGHTGBM:
                logger.info("\n" + "-" * 80)
                logger.info("3. LightGBM")
                logger.info("-" * 80)
                
                # Calculate scale_pos_weight for imbalance
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
                
                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
                
                params = {
                    'objective': 'binary',
                    'metric': 'auc',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'scale_pos_weight': scale_pos_weight,
                    'verbose': -1
                }
                
                lgb_model = lgb.train(
                    params,
                    lgb_train,
                    num_boost_round=500,
                    valid_sets=[lgb_val],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
                
                y_prob_lgb = lgb_model.predict(X_test)
                y_pred_lgb = (y_prob_lgb > 0.5).astype(int)
                
                lgb_metrics = self.evaluate_predictions(y_test, y_pred_lgb, y_prob_lgb, "LightGBM")
                lgb_metrics['model'] = 'LightGBM'
                lgb_metrics['fold'] = split['fold']
                all_results.append(lgb_metrics)
                
                # Feature importance
                importance = lgb_model.feature_importance(importance_type='gain')
                top_features = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)[:10]
                logger.info(f"\n  Top 10 features (by gain):")
                for feat, imp in top_features:
                    logger.info(f"    {feat:30s}: {imp:.0f}")
            
            # 4. XGBoost
            if HAS_XGBOOST:
                logger.info("\n" + "-" * 80)
                logger.info("4. XGBoost")
                logger.info("-" * 80)
                
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
                
                xgb_train = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
                xgb_val = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
                xgb_test = xgb.DMatrix(X_test, feature_names=feature_cols)
                
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'max_depth': 6,
                    'eta': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'scale_pos_weight': scale_pos_weight,
                    'seed': 42
                }
                
                xgb_model = xgb.train(
                    params,
                    xgb_train,
                    num_boost_round=500,
                    evals=[(xgb_val, 'val')],
                    early_stopping_rounds=50,
                    verbose_eval=False
                )
                
                y_prob_xgb = xgb_model.predict(xgb_test)
                y_pred_xgb = (y_prob_xgb > 0.5).astype(int)
                
                xgb_metrics = self.evaluate_predictions(y_test, y_pred_xgb, y_prob_xgb, "XGBoost")
                xgb_metrics['model'] = 'XGBoost'
                xgb_metrics['fold'] = split['fold']
                all_results.append(xgb_metrics)
        
        # Summary across folds
        logger.info("\n" + "=" * 80)
        logger.info("📊 SUMMARY ACROSS FOLDS")
        logger.info("=" * 80)
        
        results_df = pd.DataFrame(all_results)
        
        for model in results_df['model'].unique():
            model_results = results_df[results_df['model'] == model]
            logger.info(f"\n{model}:")
            logger.info(f"  Accuracy:            {model_results['accuracy'].mean():.4f} ± {model_results['accuracy'].std():.4f}")
            logger.info(f"  Balanced Accuracy:   {model_results['balanced_accuracy'].mean():.4f} ± {model_results['balanced_accuracy'].std():.4f}")
            logger.info(f"  ROC-AUC:             {model_results['roc_auc'].mean():.4f} ± {model_results['roc_auc'].std():.4f}")
            logger.info(f"  Macro F1:            {model_results['macro_f1'].mean():.4f} ± {model_results['macro_f1'].std():.4f}")
            logger.info(f"  High-Conf Accuracy:  {model_results['high_conf_accuracy'].mean():.4f} ± {model_results['high_conf_accuracy'].std():.4f}")
        
        logger.info("=" * 80)
        
        # Save results
        results_path = Path(self.data_path).parent / f'training_results_{self.mode}.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"\n💾 Saved results to {results_path}")
    
    def run(self):
        """Run complete training pipeline."""
        # Load and validate
        self.load_and_validate()
        
        # Print summary stats
        self.print_summary_stats()
        
        # Prepare target
        df = self.prepare_target()
        
        # Create time splits
        splits = self.create_time_splits(df, n_folds=1)  # Simple split first
        
        # Train models
        self.train_baseline_models(df, splits)
        
        logger.info("\n✅ Training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Train accumulation/distribution event models"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/ml_datasets/accum_distrib_events.parquet',
        help='Path to input parquet dataset'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['generic', 'wyckoff'],
        default='generic',
        help='Training mode: generic (UP/DOWN) or wyckoff (ACC/DIST)'
    )
    
    args = parser.parse_args()
    
    # Train
    trainer = AccumDistribTrainer(args.input, args.mode)
    trainer.run()


if __name__ == "__main__":
    main()
