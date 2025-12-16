import os
import sys
from pathlib import Path
import subprocess
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_strategy(name, env_vars):
    """Run backtest with specific environment variables."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    # Build command
    cmd = "source ML/.env.local && "
    for k, v in env_vars.items():
        cmd += f"{k}={v} "
    cmd += "python ML/backtest_pmcc_stockproxy.py 2>&1 | tail -20"
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/zsh')
    
    # Parse results
    output = result.stdout
    
    # Extract key metrics
    metrics = {"strategy": name}
    
    for line in output.split('\n'):
        if 'cagr:' in line:
            metrics['cagr'] = float(line.split(':')[1].strip())
        elif 'ann_vol:' in line:
            metrics['ann_vol'] = float(line.split(':')[1].strip())
        elif 'sharpe_like:' in line:
            metrics['sharpe'] = float(line.split(':')[1].strip())
        elif 'max_drawdown:' in line:
            metrics['max_drawdown'] = float(line.split(':')[1].strip())
        elif 'start_eq:' in line:
            metrics['start_eq'] = float(line.split(':')[1].strip())
        elif 'end_eq:' in line:
            metrics['end_eq'] = float(line.split(':')[1].strip())
    
    # Extract regime breakdown
    if 'MIDDLE' in output:
        for line in output.split('\n'):
            if 'MIDDLE' in line:
                parts = line.split()
                if len(parts) >= 3:
                    metrics['middle_ret'] = float(parts[1])
            elif 'RANGE' in line:
                parts = line.split()
                if len(parts) >= 3:
                    metrics['range_ret'] = float(parts[1])
            elif 'TREND' in line and 'SKIP' not in line:
                parts = line.split()
                if len(parts) >= 3:
                    metrics['trend_ret'] = float(parts[1])
    
    return metrics


def main():
    MODEL_VERSION = "rangeH5_xgb_sigmoid_2025-12-15"
    
    strategies = []
    
    # 1. Fixed delta strategies (baseline)
    print("\n" + "="*60)
    print("PART 1: FIXED DELTA STRATEGIES (No Regime Adaptation)")
    print("="*60)
    
    strategies.append(run_strategy(
        "Fixed Delta 0.15 (Conservative)",
        {
            "MODEL_VERSION": MODEL_VERSION,
            "DELTA_RANGE": "0.15",
            "DELTA_MIDDLE": "0.15",
            "DELTA_TREND": "0.15",
            "SKIP_TREND": "0"
        }
    ))
    
    strategies.append(run_strategy(
        "Fixed Delta 0.18 (Moderate)",
        {
            "MODEL_VERSION": MODEL_VERSION,
            "DELTA_RANGE": "0.18",
            "DELTA_MIDDLE": "0.18",
            "DELTA_TREND": "0.18",
            "SKIP_TREND": "0"
        }
    ))
    
    strategies.append(run_strategy(
        "Fixed Delta 0.25 (Aggressive)",
        {
            "MODEL_VERSION": MODEL_VERSION,
            "DELTA_RANGE": "0.25",
            "DELTA_MIDDLE": "0.25",
            "DELTA_TREND": "0.25",
            "SKIP_TREND": "0"
        }
    ))
    
    # 2. Regime-adaptive strategies
    print("\n" + "="*60)
    print("PART 2: REGIME-ADAPTIVE STRATEGIES")
    print("="*60)
    
    strategies.append(run_strategy(
        "Regime-Adaptive: Conservative TREND",
        {
            "MODEL_VERSION": MODEL_VERSION,
            "DELTA_RANGE": "0.25",
            "DELTA_MIDDLE": "0.18",
            "DELTA_TREND": "0.10",  # Conservative in trend
            "SKIP_TREND": "0"
        }
    ))
    
    strategies.append(run_strategy(
        "Regime-Adaptive: Skip TREND",
        {
            "MODEL_VERSION": MODEL_VERSION,
            "DELTA_RANGE": "0.25",
            "DELTA_MIDDLE": "0.18",
            "DELTA_TREND": "0.10",
            "SKIP_TREND": "1"  # Don't write calls in trend
        }
    ))
    
    strategies.append(run_strategy(
        "Regime-Adaptive: Aggressive RANGE",
        {
            "MODEL_VERSION": MODEL_VERSION,
            "DELTA_RANGE": "0.30",  # Very aggressive in range
            "DELTA_MIDDLE": "0.18",
            "DELTA_TREND": "0.12",
            "SKIP_TREND": "0"
        }
    ))
    
    # 3. Alternative regime interpretations
    print("\n" + "="*60)
    print("PART 3: ALTERNATIVE REGIME STRATEGIES")
    print("="*60)
    
    strategies.append(run_strategy(
        "Inverse: Aggressive TREND, Conservative RANGE",
        {
            "MODEL_VERSION": MODEL_VERSION,
            "DELTA_RANGE": "0.12",  # Conservative in range
            "DELTA_MIDDLE": "0.18",
            "DELTA_TREND": "0.28",  # Aggressive in trend (contrarian)
            "SKIP_TREND": "0"
        }
    ))
    
    strategies.append(run_strategy(
        "Equal Weight with TREND Skip",
        {
            "MODEL_VERSION": MODEL_VERSION,
            "DELTA_RANGE": "0.20",
            "DELTA_MIDDLE": "0.20",
            "DELTA_TREND": "0.20",
            "SKIP_TREND": "1"  # Skip trend periods entirely
        }
    ))
    
    # Create comparison DataFrame
    df = pd.DataFrame(strategies)
    
    # Calculate additional metrics
    if 'start_eq' in df.columns and 'end_eq' in df.columns:
        df['total_return'] = (df['end_eq'] / df['start_eq'] - 1) * 100
    
    if 'sharpe' in df.columns and 'ann_vol' in df.columns:
        df['return_per_vol'] = df['cagr'] / df['ann_vol']
    
    # Sort by Sharpe ratio
    df_sorted = df.sort_values('sharpe', ascending=False)
    
    # Display results
    print("\n" + "="*60)
    print("STRATEGY COMPARISON RESULTS")
    print("="*60)
    
    print("\n### Overall Performance:")
    display_cols = ['strategy', 'cagr', 'ann_vol', 'sharpe', 'max_drawdown', 'total_return']
    available_cols = [c for c in display_cols if c in df_sorted.columns]
    print(df_sorted[available_cols].to_string(index=False))
    
    print("\n### Regime-Specific Returns:")
    regime_cols = ['strategy', 'range_ret', 'middle_ret', 'trend_ret']
    available_regime_cols = [c for c in regime_cols if c in df_sorted.columns]
    if len(available_regime_cols) > 1:
        print(df_sorted[available_regime_cols].to_string(index=False))
    
    # Save to CSV
    output_file = "ML/strategy_comparison.csv"
    df_sorted.to_csv(output_file, index=False)
    print(f"\n✅ Saved detailed results to: {output_file}")
    
    # Find best strategies
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    best_sharpe = df_sorted.iloc[0]
    print(f"\n🏆 Best Sharpe Ratio: {best_sharpe['strategy']}")
    print(f"   Sharpe: {best_sharpe['sharpe']:.4f}, CAGR: {best_sharpe['cagr']:.2%}")
    
    if 'total_return' in df_sorted.columns:
        best_return = df_sorted.nlargest(1, 'total_return').iloc[0]
        print(f"\n💰 Highest Total Return: {best_return['strategy']}")
        print(f"   Return: {best_return['total_return']:.2f}%, Sharpe: {best_return['sharpe']:.4f}")
    
    # Check if regime adaptation helps
    fixed_strategies = df_sorted[df_sorted['strategy'].str.contains('Fixed')]
    adaptive_strategies = df_sorted[df_sorted['strategy'].str.contains('Regime-Adaptive')]
    
    if len(fixed_strategies) > 0 and len(adaptive_strategies) > 0:
        avg_fixed_sharpe = fixed_strategies['sharpe'].mean()
        avg_adaptive_sharpe = adaptive_strategies['sharpe'].mean()
        
        print(f"\n📊 Regime Adaptation Impact:")
        print(f"   Avg Fixed Sharpe: {avg_fixed_sharpe:.4f}")
        print(f"   Avg Adaptive Sharpe: {avg_adaptive_sharpe:.4f}")
        print(f"   Improvement: {((avg_adaptive_sharpe / avg_fixed_sharpe - 1) * 100):.1f}%")


if __name__ == "__main__":
    main()
