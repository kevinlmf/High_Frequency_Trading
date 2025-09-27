#!/usr/bin/env python3
"""
HFT Regime-Based Statistical Approach Comparison
===============================================

Main script for running comprehensive analysis comparing Frequentist vs Bayesian
approaches across 9 HFT market regimes (Liquidity × Volume × Volatility).

This script integrates:
1. Market regime classification
2. Frequentist statistical methods (GARCH, Hawkes, classical tests)
3. Bayesian statistical methods (hierarchical, switching, dynamic)
4. Comprehensive comparison and visualization

Usage:
    python run_regime_statistical_comparison.py --symbol AAPL --period 5d --interval 1m
    python run_regime_statistical_comparison.py --symbol BTCUSDT --quick --generate-report
    python run_regime_statistical_comparison.py --file data/custom_data.csv --full-analysis
"""

import argparse
import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Core imports
from regime_comparison.statistical_approach_comparator import StatisticalApproachComparator
from signal_engine.data_sources.yahoo_finance import YahooFinanceSource


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='HFT Regime Statistical Comparison Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with Yahoo Finance data
  python run_regime_statistical_comparison.py --symbol AAPL --period 30d --interval 5m

  # Quick analysis (faster, less comprehensive)
  python run_regime_statistical_comparison.py --symbol MSFT --quick

  # Full analysis with PDF report generation
  python run_regime_statistical_comparison.py --symbol TSLA --period 10d --full-analysis --generate-report

  # Use custom data file
  python run_regime_statistical_comparison.py --file data/custom_ohlcv.csv --generate-report

  # Compare multiple timeframes
  python run_regime_statistical_comparison.py --symbol BTC-USD --period 7d --interval 1m --export-csv
        """
    )

    # Data source options
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--symbol', type=str,
                           help='Stock/crypto symbol (e.g., AAPL, BTC-USD)')
    data_group.add_argument('--file', type=str,
                           help='Path to custom CSV file with OHLCV data')

    # Yahoo Finance options (when using --symbol)
    parser.add_argument('--period', type=str, default='5d',
                       choices=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
                       help='Period to download (default: 5d)')
    parser.add_argument('--interval', type=str, default='5m',
                       choices=['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'],
                       help='Data interval (default: 5m)')

    # Analysis options
    parser.add_argument('--quick', action='store_true',
                       help='Quick analysis (reduced computational complexity)')
    parser.add_argument('--full-analysis', action='store_true',
                       help='Full comprehensive analysis (more detailed but slower)')

    # Output options
    parser.add_argument('--output-dir', type=str, default='exports',
                       help='Output directory for results (default: exports)')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate comprehensive PDF report')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export results to CSV format')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save visualization plots')

    # Advanced options
    parser.add_argument('--regime-params', type=str,
                       help='JSON string with custom regime classification parameters')
    parser.add_argument('--min-regime-samples', type=int, default=20,
                       help='Minimum samples required per regime (default: 20)')

    return parser.parse_args()


def load_data(args):
    """
    Load market data based on arguments

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (DataFrame, symbol_name)
    """
    if args.symbol:
        logger.info(f"Downloading data for {args.symbol} (period: {args.period}, interval: {args.interval})")

        try:
            yahoo_source = YahooFinanceSource()
            data = yahoo_source.get_data(
                symbol=args.symbol,
                period=args.period,
                interval=args.interval
            )

            if data is None or len(data) == 0:
                raise ValueError("No data received from Yahoo Finance")

            logger.info(f"Downloaded {len(data)} data points for {args.symbol}")
            return data, args.symbol

        except Exception as e:
            logger.error(f"Failed to download data for {args.symbol}: {str(e)}")
            raise

    elif args.file:
        logger.info(f"Loading data from file: {args.file}")

        try:
            # Try to load CSV file
            data = pd.read_csv(args.file, index_col=0, parse_dates=True)

            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Convert to lowercase column names (standard format)
            data.columns = data.columns.str.lower()

            symbol_name = Path(args.file).stem
            logger.info(f"Loaded {len(data)} data points from {args.file}")
            return data, symbol_name

        except Exception as e:
            logger.error(f"Failed to load data from {args.file}: {str(e)}")
            raise

    else:
        raise ValueError("Either --symbol or --file must be specified")


def validate_data(data: pd.DataFrame, min_samples: int = 100) -> bool:
    """
    Validate data quality and sufficiency

    Args:
        data: Market data DataFrame
        min_samples: Minimum required samples

    Returns:
        True if data is valid
    """
    if len(data) < min_samples:
        logger.warning(f"Data has only {len(data)} samples, minimum recommended: {min_samples}")
        return False

    # Check for missing values
    missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
    if missing_pct > 5:
        logger.warning(f"Data has {missing_pct:.1f}% missing values")

    # Check for valid price ranges
    if (data['close'] <= 0).any():
        logger.error("Invalid price data: found zero or negative prices")
        return False

    if (data['volume'] < 0).any():
        logger.error("Invalid volume data: found negative volumes")
        return False

    logger.info("Data validation passed")
    return True


def setup_comparator(args):
    """
    Setup the statistical approach comparator with appropriate parameters

    Args:
        args: Parsed command line arguments

    Returns:
        Configured StatisticalApproachComparator
    """
    # Base parameters for regime classifier
    regime_params = {
        'lookback_period': 50 if args.quick else 100,
        'liquidity_thresholds': (0.33, 0.67),
        'volume_threshold': 0.5,
        'volatility_threshold': 0.5
    }

    # Update with custom parameters if provided
    if args.regime_params:
        import json
        try:
            custom_params = json.loads(args.regime_params)
            regime_params.update(custom_params)
            logger.info(f"Using custom regime parameters: {custom_params}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse custom regime parameters: {e}")

    comparator = StatisticalApproachComparator(regime_classifier_params=regime_params)

    logger.info("Statistical approach comparator initialized")
    return comparator


def run_analysis(comparator, data, symbol, args):
    """
    Run the comprehensive statistical comparison analysis

    Args:
        comparator: Configured comparator
        data: Market data
        symbol: Symbol identifier
        args: Command line arguments

    Returns:
        Analysis results dictionary
    """
    logger.info(f"Starting comprehensive regime analysis for {symbol}")
    logger.info(f"Analysis scope: {'Quick' if args.quick else 'Full comprehensive' if args.full_analysis else 'Standard'}")

    start_time = time.time()

    try:
        # Run comprehensive comparison
        results = comparator.run_comprehensive_comparison(data, symbol)

        analysis_time = time.time() - start_time
        logger.info(f"Analysis completed in {analysis_time:.2f} seconds")

        # Add runtime metadata
        results['analysis_metadata'] = {
            'runtime_seconds': analysis_time,
            'analysis_type': 'quick' if args.quick else 'full' if args.full_analysis else 'standard',
            'data_points': len(data),
            'date_range': f"{data.index[0]} to {data.index[-1]}",
            'parameters_used': {
                'min_regime_samples': args.min_regime_samples,
                'interval': getattr(args, 'interval', 'unknown'),
                'period': getattr(args, 'period', 'unknown')
            }
        }

        return results

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


def generate_outputs(results, comparator, symbol, args):
    """
    Generate various output formats based on arguments

    Args:
        results: Analysis results
        comparator: Comparator instance
        symbol: Symbol name
        args: Command line arguments
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    generated_files = []

    # 1. CSV Export
    if args.export_csv or args.full_analysis:
        logger.info("Exporting results to CSV...")
        try:
            csv_path = comparator.export_results_to_csv(results, str(output_dir))
            if csv_path:
                generated_files.append(csv_path)
                logger.info(f"✅ CSV export completed: {csv_path}")
        except Exception as e:
            logger.error(f"CSV export failed: {str(e)}")

    # 2. Visualization plots
    if args.save_plots or args.full_analysis:
        logger.info("Generating visualization plots...")
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plot_path = output_dir / f"{symbol}_regime_analysis_{timestamp}.png"

            fig = comparator.generate_regime_matrix_visualization(results, str(plot_path))
            if fig is not None:
                generated_files.append(str(plot_path))
                logger.info(f"✅ Visualization saved: {plot_path}")
                plt.close(fig)  # Free memory
        except Exception as e:
            logger.error(f"Plot generation failed: {str(e)}")

    # 3. Text summary report
    logger.info("Generating text summary report...")
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary_path = output_dir / f"{symbol}_regime_summary_{timestamp}.txt"

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(generate_text_report(results, symbol))

        generated_files.append(str(summary_path))
        logger.info(f"✅ Text summary saved: {summary_path}")

    except Exception as e:
        logger.error(f"Text summary generation failed: {str(e)}")

    # 4. JSON results (for programmatic access)
    if args.full_analysis:
        logger.info("Exporting detailed JSON results...")
        try:
            import json
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            json_path = output_dir / f"{symbol}_regime_results_{timestamp}.json"

            # Convert numpy types for JSON serialization
            json_results = convert_for_json(results)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)

            generated_files.append(str(json_path))
            logger.info(f"✅ JSON export completed: {json_path}")

        except Exception as e:
            logger.error(f"JSON export failed: {str(e)}")

    return generated_files


def generate_text_report(results, symbol):
    """
    Generate comprehensive text report

    Args:
        results: Analysis results
        symbol: Symbol name

    Returns:
        Formatted text report
    """
    report = []
    report.append("=" * 80)
    report.append(f"HFT REGIME STATISTICAL COMPARISON REPORT")
    report.append(f"Symbol: {symbol}")
    report.append(f"Analysis Date: {results.get('analysis_date', 'Unknown')}")
    report.append("=" * 80)
    report.append("")

    # Executive Summary
    summary = results.get('summary_analysis', {})
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 20)
    report.append(f"Total Regimes Analyzed: {summary.get('total_regimes_analyzed', 0)}")
    report.append(f"Theory Validation Rate: {summary.get('theory_match_rate', 0):.1%}")
    report.append("")

    empirical = summary.get('empirical_results', {})
    report.append("Empirical Results:")
    report.append(f"  • Frequentist Wins: {empirical.get('frequentist_wins', 0)}")
    report.append(f"  • Bayesian Wins: {empirical.get('bayesian_wins', 0)}")
    report.append(f"  • Ties: {empirical.get('ties', 0)}")
    report.append("")

    overall_rec = summary.get('overall_recommendation', 'No recommendation available')
    report.append(f"Overall Recommendation: {overall_rec}")
    report.append("")

    # Key Insights
    insights = results.get('key_insights', [])
    if insights:
        report.append("KEY INSIGHTS")
        report.append("-" * 15)
        for insight in insights:
            report.append(f"• {insight}")
        report.append("")

    # Regime-by-Regime Analysis
    individual_results = results.get('individual_results', [])
    if individual_results:
        report.append("DETAILED REGIME ANALYSIS")
        report.append("-" * 28)

        for i, result in enumerate(individual_results, 1):
            report.append(f"\n{i}. {result.regime_name}")
            report.append(f"   Characteristics: {result.regime_characteristics}")
            report.append(f"   Sample Size: {result.sample_size}")
            report.append(f"   Theoretical Winner: {result.theoretical_winner}")
            report.append(f"   Empirical Winner: {result.empirical_winner}")
            report.append(f"   Scores: Freq={result.frequentist_score:.3f}, Bayes={result.bayesian_score:.3f}")
            report.append(f"   Theory Match: {'✓' if result.matches_theory else '✗'}")

            if result.key_insights:
                report.append(f"   Key Insights:")
                for insight in result.key_insights[:2]:  # Limit for brevity
                    report.append(f"     - {insight}")

    # 9-Grid Theory Validation Matrix
    report.append("\n")
    report.append("THEORETICAL FRAMEWORK VALIDATION")
    report.append("-" * 40)
    report.append("Based on your original 9-grid hypothesis:")
    report.append("")

    grid_explanation = [
        "High Liquidity × High Volume × High Volatility → Bayesian (uncertainty critical)",
        "High Liquidity × High Volume × Low Volatility → Frequentist (efficient)",
        "High Liquidity × Low Volume × High Volatility → Bayesian (sparse data)",
        "Low Liquidity × High Volume × High Volatility → Bayesian (regime shifts)",
        "Low Liquidity × Low Volume × High Volatility → Bayesian (data scarce)",
        "Mixed regimes → Hybrid approach"
    ]

    for explanation in grid_explanation:
        report.append(f"• {explanation}")

    # Methodology Notes
    methodology = results.get('methodology_notes', {})
    if methodology:
        report.append("\n")
        report.append("METHODOLOGY NOTES")
        report.append("-" * 20)
        for key, value in methodology.items():
            report.append(f"• {key.title()}: {value}")

    # Runtime Information
    metadata = results.get('analysis_metadata', {})
    if metadata:
        report.append("\n")
        report.append("ANALYSIS METADATA")
        report.append("-" * 20)
        runtime = metadata.get('runtime_seconds', 0)
        report.append(f"• Analysis Runtime: {runtime:.2f} seconds")
        report.append(f"• Data Points Analyzed: {metadata.get('data_points', 'Unknown')}")
        report.append(f"• Date Range: {metadata.get('date_range', 'Unknown')}")

    report.append("\n")
    report.append("=" * 80)
    report.append("Report generated by HFT Regime Statistical Comparison System")
    report.append("=" * 80)

    return "\n".join(report)


def convert_for_json(obj):
    """Convert numpy types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj


def print_summary_to_console(results, symbol):
    """Print concise summary to console"""
    print("\n" + "="*60)
    print(f"🎯 HFT REGIME ANALYSIS COMPLETE: {symbol}")
    print("="*60)

    summary = results.get('summary_analysis', {})
    regimes_analyzed = summary.get('total_regimes_analyzed', 0)
    theory_match_rate = summary.get('theory_match_rate', 0)

    print(f"📊 Regimes Analyzed: {regimes_analyzed}")
    print(f"🎯 Theory Validation: {theory_match_rate:.1%}")

    empirical = summary.get('empirical_results', {})
    freq_wins = empirical.get('frequentist_wins', 0)
    bayes_wins = empirical.get('bayesian_wins', 0)
    ties = empirical.get('ties', 0)

    print(f"🏆 Results: Freq={freq_wins}, Bayes={bayes_wins}, Ties={ties}")

    overall_rec = summary.get('overall_recommendation', '')
    print(f"💡 Recommendation: {overall_rec}")

    # Key insights
    insights = results.get('key_insights', [])
    if insights:
        print(f"\n🔍 Key Insights:")
        for insight in insights[:3]:  # Show top 3
            print(f"   • {insight}")

    metadata = results.get('analysis_metadata', {})
    runtime = metadata.get('runtime_seconds', 0)
    print(f"\n⏱️  Analysis completed in {runtime:.2f} seconds")
    print("="*60 + "\n")


def main():
    """Main execution function"""
    try:
        # Parse arguments
        args = parse_arguments()
        logger.info("Starting HFT Regime Statistical Comparison Analysis")

        # Load data
        data, symbol = load_data(args)

        # Validate data
        if not validate_data(data, args.min_regime_samples):
            logger.error("Data validation failed - proceeding with caution")

        # Setup comparator
        comparator = setup_comparator(args)

        # Run analysis
        results = run_analysis(comparator, data, symbol, args)

        # Print console summary
        print_summary_to_console(results, symbol)

        # Generate outputs
        generated_files = generate_outputs(results, comparator, symbol, args)

        # Final summary
        if generated_files:
            print("📁 Generated Files:")
            for file_path in generated_files:
                print(f"   • {file_path}")

        print(f"\n✅ Analysis complete! Check the {args.output_dir} directory for detailed results.")

        logger.info("HFT Regime Statistical Comparison completed successfully")

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        print("\n⚠️  Analysis interrupted by user")

    except Exception as e:
        logger.error(f"Analysis failed with error: {str(e)}")
        print(f"\n❌ Analysis failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())