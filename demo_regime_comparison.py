#!/usr/bin/env python3
"""
Demo Script: HFT Regime Statistical Comparison
==============================================

Simple demonstration of the HFT regime-based statistical approach comparison system.
This script shows how to use the system with minimal setup.

Usage:
    python demo_regime_comparison.py
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our system components
from regime_comparison.statistical_approach_comparator import StatisticalApproachComparator


def generate_synthetic_hft_data(n_points: int = 1000, regime_changes: int = 3) -> pd.DataFrame:
    """
    Generate synthetic HFT-like data with different regime characteristics

    Args:
        n_points: Number of data points to generate
        regime_changes: Number of regime changes to include

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Generating synthetic HFT data with {n_points} points and {regime_changes} regime changes")

    # Create date range (every 5 minutes for realistic HFT timestamps)
    start_date = datetime.now() - timedelta(days=7)
    timestamps = pd.date_range(start=start_date, periods=n_points, freq='5T')

    # Initialize arrays
    prices = np.zeros(n_points)
    volumes = np.zeros(n_points)

    # Starting values
    base_price = 100.0
    current_price = base_price

    # Create regime segments
    regime_lengths = np.diff(np.concatenate([[0],
                                           np.sort(np.random.choice(n_points, regime_changes, replace=False)),
                                           [n_points]]))

    # Define regime characteristics (Liquidity, Volume, Volatility)
    regime_types = [
        {'name': 'HighLiq_HighVol_HighVol', 'spread': 0.001, 'vol_mult': 2.0, 'volatility': 0.02},
        {'name': 'HighLiq_HighVol_LowVol', 'spread': 0.001, 'vol_mult': 2.0, 'volatility': 0.005},
        {'name': 'LowLiq_LowVol_HighVol', 'spread': 0.01, 'vol_mult': 0.3, 'volatility': 0.03},
        {'name': 'MedLiq_HighVol_HighVol', 'spread': 0.005, 'vol_mult': 1.5, 'volatility': 0.015},
    ]

    idx = 0
    for i, length in enumerate(regime_lengths):
        # Select regime type
        regime = regime_types[i % len(regime_types)]
        logger.info(f"Generating regime {regime['name']} for {length} points")

        for j in range(length):
            if idx >= n_points:
                break

            # Generate price with regime-specific characteristics
            price_change = np.random.normal(0, regime['volatility']) * current_price
            current_price += price_change
            current_price = max(current_price, base_price * 0.5)  # Floor price

            # Generate OHLC from current price
            spread = regime['spread'] * current_price
            high = current_price + np.random.uniform(0, spread * 2)
            low = current_price - np.random.uniform(0, spread * 2)
            open_price = current_price + np.random.uniform(-spread, spread)

            prices[idx] = current_price

            # Generate volume with regime characteristics
            base_volume = 1000
            volume_noise = np.random.lognormal(0, 0.5)  # Log-normal for realistic volume distribution
            volumes[idx] = base_volume * regime['vol_mult'] * volume_noise

            idx += 1

    # Create OHLC data
    data = pd.DataFrame(index=timestamps)

    # Simple OHLC generation from close prices
    data['close'] = prices
    data['open'] = np.roll(prices, 1)  # Previous close as open
    data['open'][0] = prices[0]

    # Add some noise to create realistic high/low
    noise_pct = 0.002  # 0.2% noise
    data['high'] = data['close'] * (1 + np.abs(np.random.normal(0, noise_pct, n_points)))
    data['low'] = data['close'] * (1 - np.abs(np.random.normal(0, noise_pct, n_points)))

    # Ensure high >= close >= low
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])

    data['volume'] = volumes

    logger.info("Synthetic data generation completed")
    return data


def run_demo():
    """Run the demo analysis"""

    print("HFT Regime Statistical Comparison Demo")
    print("=" * 50)

    # Step 1: Generate synthetic data
    print("\nStep 1: Generating synthetic HFT data...")
    data = generate_synthetic_hft_data(n_points=800, regime_changes=4)
    print(f"Generated {len(data)} data points from {data.index[0]} to {data.index[-1]}")

    # Quick data preview
    print(f"\nData Preview:")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"   Volume range: {data['volume'].min():.0f} - {data['volume'].max():.0f}")
    print(f"   Returns std: {data['close'].pct_change().std():.4f}")

    # Step 2: Setup comparator
    print("\nStep 2: Setting up regime comparator...")
    comparator = StatisticalApproachComparator(
        regime_classifier_params={
            'lookback_period': 50,  # Smaller for demo
            'liquidity_thresholds': (0.3, 0.7),
            'volume_threshold': 0.5,
            'volatility_threshold': 0.5
        }
    )
    print("Comparator initialized")

    # Step 3: Run comprehensive analysis
    print("\nStep 3: Running regime statistical comparison...")
    print("   This may take a moment as we analyze all statistical methods...")

    try:
        start_time = datetime.now()

        results = comparator.run_comprehensive_comparison(data, symbol="DEMO_HFT")

        end_time = datetime.now()
        analysis_time = (end_time - start_time).total_seconds()

        print(f"Analysis completed in {analysis_time:.2f} seconds")

    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        logger.error(f"Demo analysis failed: {str(e)}")
        return

    # Step 4: Display results
    print("\nStep 4: Analysis Results")
    print("=" * 30)

    # Summary statistics
    summary = results.get('summary_analysis', {})
    regimes_found = summary.get('total_regimes_analyzed', 0)
    theory_match_rate = summary.get('theory_match_rate', 0)

    print(f"🔢 Regimes Identified: {regimes_found}")
    print(f"🎯 Theory Validation Rate: {theory_match_rate:.1%}")

    # Winners breakdown
    empirical = summary.get('empirical_results', {})
    freq_wins = empirical.get('frequentist_wins', 0)
    bayes_wins = empirical.get('bayesian_wins', 0)
    ties = empirical.get('ties', 0)

    print(f"\n🏆 Method Performance:")
    print(f"   • Frequentist wins: {freq_wins}")
    print(f"   • Bayesian wins: {bayes_wins}")
    print(f"   • Ties: {ties}")

    # Overall recommendation
    overall_rec = summary.get('overall_recommendation', 'No recommendation')
    print(f"\n💡 Overall Recommendation:")
    print(f"   {overall_rec}")

    # Key insights
    insights = results.get('key_insights', [])
    if insights:
        print(f"\n🔍 Key Insights:")
        for i, insight in enumerate(insights[:5], 1):  # Show top 5
            print(f"   {i}. {insight}")

    # Regime-by-regime breakdown
    individual_results = results.get('individual_results', [])
    if individual_results:
        print(f"\n📋 Regime Breakdown:")
        print("   " + "-" * 70)

        for result in individual_results:
            winner = result.empirical_winner
            theory_match = "✓" if result.matches_theory else "✗"

            print(f"   📍 {result.regime_name}")
            print(f"       Samples: {result.sample_size:4d} | Winner: {winner:12s} | Theory: {theory_match}")
            print(f"       Scores: Freq={result.frequentist_score:.3f}, Bayes={result.bayesian_score:.3f}")

            if result.key_insights:
                main_insight = result.key_insights[0].replace('📊', '').replace('🎯', '').replace('✅', '').strip()
                print(f"       Insight: {main_insight[:60]}...")
            print()

    # Step 5: Generate visualization (if possible)
    print("📊 Step 5: Generating visualization...")
    try:
        import matplotlib.pyplot as plt
        fig = comparator.generate_regime_matrix_visualization(results, save_path="demo_regime_analysis.png")
        if fig:
            print("✅ Visualization saved as 'demo_regime_analysis.png'")
            plt.close(fig)
        else:
            print("⚠️  Visualization generation skipped (insufficient data)")
    except Exception as e:
        print(f"⚠️  Visualization failed: {str(e)}")

    # Step 6: Export results
    print("\n💾 Step 6: Exporting results...")
    try:
        csv_path = comparator.export_results_to_csv(results, output_dir="demo_output")
        if csv_path:
            print(f"✅ Results exported to: {csv_path}")
        else:
            print("⚠️  CSV export skipped")
    except Exception as e:
        print(f"⚠️  Export failed: {str(e)}")

    # Demo conclusion
    print("\n" + "=" * 50)
    print("🎉 DEMO COMPLETE!")
    print("=" * 50)

    print("\n🎯 What this demo showed:")
    print("   • Automatic market regime classification")
    print("   • Frequentist vs Bayesian method comparison")
    print("   • Theory validation against empirical results")
    print("   • Regime-specific statistical approach recommendations")

    print("\n🚀 Next steps:")
    print("   • Try with real market data using: python run_regime_statistical_comparison.py --symbol AAPL --period 5d")
    print("   • Experiment with different regime classification parameters")
    print("   • Use the system for your own HFT strategy development")

    print("\n📚 Your 9-Grid Theory Validation:")
    if theory_match_rate > 0.7:
        print("   ✅ Strong validation - your theoretical framework holds well!")
    elif theory_match_rate > 0.5:
        print("   ⚠️  Partial validation - some regimes behave differently than expected")
    else:
        print("   🔍 Weak validation - consider refining the theoretical framework")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    run_demo()