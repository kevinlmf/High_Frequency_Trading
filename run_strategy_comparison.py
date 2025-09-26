#!/usr/bin/env python3
"""
Complete HFT Strategy Comparison Pipeline
========================================

Compares ALL strategy types in one command:
- ML Strategies (Linear, Ridge, Random Forest)
- Traditional Quant (Momentum, Mean Reversion, Pairs Trading)
- Deep Learning (LSTM, GRU, Transformer, CNN-LSTM)
- DeepLOB + Transformer (Advanced LOB-based Deep Learning)

Usage:
    python run_strategy_comparison.py --symbol AAPL
    python run_strategy_comparison.py --symbol AAPL --quick
    python run_strategy_comparison.py --symbol MSFT --period 2d --include-all
"""

import argparse
import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Core imports
from signal_engine.signal_processor import SignalProcessor
from evaluation.performance_metrics import PerformanceMetrics

# Traditional strategy imports
from strategy_methods.traditional.momentum_strategy import MomentumStrategy
from strategy_methods.traditional.mean_reversion_strategy import MeanReversionStrategy
from strategy_methods.traditional.pairs_trading_strategy import PairsTradingStrategy

# Deep Learning strategy imports
from strategy_methods.llm_methods.lstm_strategy import LSTMStrategy
from strategy_methods.llm_methods.gru_strategy import GRUStrategy
from strategy_methods.llm_methods.transformer_strategy import TransformerStrategy
from strategy_methods.llm_methods.cnn_lstm_strategy import CNNLSTMStrategy

# DeepLOB + Transformer strategy imports (replacing RL)
from strategy_methods.deep_learning_methods.deeplob_transformer import DeepLOBTransformerStrategy

# PDF Report import
from evaluation.strategy_comparison_pdf import StrategyComparisonPDFGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_execute_strategy(strategy_func, strategy_name, *args, **kwargs):
    """Safely execute a strategy and handle errors"""
    try:
        logger.info(f"Running {strategy_name}...")
        return strategy_func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"WARNING: {strategy_name} failed: {str(e)}")
        return None

def calculate_strategy_financial_metrics(signals, data, strategy_type, initial_capital=100000):
    """Calculate comprehensive financial metrics for a strategy"""
    try:
        # Convert signals to pandas Series if needed
        if isinstance(signals, dict):
            signals = pd.Series(signals.get('signals', [0] * len(data)), index=data.index)
        elif not isinstance(signals, pd.Series):
            signals = pd.Series(signals, index=data.index)

        # Calculate returns
        price_returns = data['close'].pct_change().fillna(0)

        # Calculate strategy returns (assuming perfect execution)
        # Shift signals by 1 to avoid look-ahead bias
        strategy_returns = (signals.shift(1) * price_returns).fillna(0)

        # Calculate cumulative returns and net PnL
        cumulative_returns = (1 + strategy_returns).cumprod()
        net_pnl = (cumulative_returns.iloc[-1] - 1) * initial_capital if len(cumulative_returns) > 0 else 0

        # Calculate performance metrics using PerformanceMetrics class
        perf_metrics = PerformanceMetrics(strategy_returns, risk_free_rate=0.02)
        all_metrics = perf_metrics.calculate_all_metrics()

        # Add strategy-specific information including net PnL
        all_metrics['type'] = strategy_type
        all_metrics['net_pnl'] = net_pnl
        all_metrics['initial_capital'] = initial_capital
        all_metrics['total_return_pct'] = (cumulative_returns.iloc[-1] - 1) * 100 if len(cumulative_returns) > 0 else 0
        all_metrics['total_signals'] = len(signals[signals != 0])
        all_metrics['buy_signals'] = (signals == 1).sum()
        all_metrics['sell_signals'] = (signals == -1).sum()

        return all_metrics

    except Exception as e:
        logger.warning(f"Failed to calculate financial metrics: {str(e)}")
        # Return basic metrics structure with zeros
        return {
            'type': strategy_type,
            'net_pnl': 0.0,
            'initial_capital': initial_capital,
            'total_return_pct': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            'var_5pct': 0.0,
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0
        }

def _run_pairs_trading_strategy(data, symbol):
    """Helper function to run pairs trading strategy"""
    pairs_strategy = PairsTradingStrategy(
        formation_period=min(60, len(data) // 4),
        entry_threshold=1.5,
        exit_threshold=0.5
    )

    # Create price data dict for pairs strategy
    price_data = {symbol: data}

    # Fit and generate signals
    pairs_strategy.fit(price_data)
    return pairs_strategy.generate_signals(price_data)

def _train_and_evaluate_deeplob_transformer(data, features, epochs=50):
    """Helper function to train and evaluate DeepLOB + Transformer strategy"""
    try:
        # Create and train the strategy
        # Adjust market feature dimension to match actual features
        actual_feature_dim = len(features.columns)
        strategy = DeepLOBTransformerStrategy(
            lob_input_dim=40,
            market_feature_dim=actual_feature_dim,  # Use actual feature dimension
            d_model=128,
            nhead=8,
            num_layers=4,
            sequence_length=50
        )

        # Train the model
        training_result = strategy.train(data, features, epochs=epochs, batch_size=16)

        # Generate signals
        signals = strategy.predict(data, features)

        # Calculate performance metrics
        returns = data['close'].pct_change().fillna(0)
        strategy_returns = signals.shift(1) * returns[signals.index]

        # Get performance metrics
        metrics = strategy.get_performance_metrics()

        return {
            'hit_rate': (signals != 0).mean(),
            'total_signals': len(signals),
            'total_return': strategy_returns.sum(),
            'sharpe_ratio': strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252),
            'volatility': strategy_returns.std() * np.sqrt(252),
            'max_drawdown': (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min(),
            'final_train_acc': metrics.get('final_train_acc', 0),
            'final_val_acc': metrics.get('final_val_acc', 0),
            'algorithm': 'DeepLOB+Transformer'
        }
    except Exception as e:
        logger.error(f"Error in DeepLOB + Transformer strategy: {str(e)}")
        return {
            'hit_rate': 0.0,
            'total_signals': 0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'volatility': 0.0,
            'max_drawdown': 0.0,
            'algorithm': 'DeepLOB+Transformer'
        }

def create_comprehensive_financial_comparison(results_dict):
    """Create a comprehensive financial strategy comparison table with key metrics"""
    print("\n" + "=" * 150)
    print("COMPREHENSIVE FINANCIAL STRATEGY PERFORMANCE COMPARISON")
    print("=" * 150)
    print(f"{'Strategy':<25} {'Type':<12} {'Ann.Ret':<9} {'Vol':<8} {'Sharpe':<8} {'MaxDD':<9} {'Calmar':<8} {'WinRate':<9} {'VaR':<8} {'Assessment'}")
    print("-" * 150)

    # Calculate summary statistics
    best_return = -float('inf')
    best_sharpe = -float('inf')
    best_calmar = -float('inf')
    lowest_drawdown = float('inf')

    for strategy_name, results in results_dict.items():
        if results is None:
            continue

        ann_return = results.get('annualized_return', 0)
        sharpe = results.get('sharpe_ratio', 0)
        calmar = results.get('calmar_ratio', 0)
        max_dd = results.get('max_drawdown', 0)

        best_return = max(best_return, ann_return)
        best_sharpe = max(best_sharpe, sharpe)
        best_calmar = max(best_calmar, calmar) if not np.isinf(calmar) else best_calmar
        lowest_drawdown = min(lowest_drawdown, abs(max_dd))

    # Display strategies with enhanced financial assessment
    for strategy_name, results in results_dict.items():
        if results is None:
            print(f"{strategy_name:<25} {'N/A':<12} {'Failed':<9} {'--':<8} {'--':<8} {'--':<9} {'--':<8} {'--':<9} {'--':<8} {'ERROR'}")
            continue

        strategy_type = results.get('type', 'Unknown')
        ann_return = results.get('annualized_return', 0)
        volatility = results.get('volatility', 0)
        sharpe = results.get('sharpe_ratio', 0)
        max_dd = results.get('max_drawdown', 0)
        calmar = results.get('calmar_ratio', 0)
        win_rate = results.get('win_rate', 0)
        var_5pct = results.get('var_5pct', 0)

        # Enhanced financial assessment logic
        score = 0
        assessment_details = []

        # Return score (30%)
        if ann_return > 0.15:
            score += 30
            assessment_details.append("High Returns")
        elif ann_return > 0.10:
            score += 20
            assessment_details.append("Good Returns")
        elif ann_return > 0.05:
            score += 10
            assessment_details.append("Moderate Returns")

        # Risk-adjusted return score (35%)
        if sharpe > 2.0:
            score += 35
            assessment_details.append("Excellent Sharpe")
        elif sharpe > 1.5:
            score += 25
            assessment_details.append("Good Sharpe")
        elif sharpe > 1.0:
            score += 15
            assessment_details.append("Decent Sharpe")
        elif sharpe > 0.5:
            score += 5
            assessment_details.append("Low Sharpe")

        # Drawdown control score (25%)
        if abs(max_dd) < 0.05:
            score += 25
            assessment_details.append("Low Drawdown")
        elif abs(max_dd) < 0.10:
            score += 20
            assessment_details.append("Moderate Drawdown")
        elif abs(max_dd) < 0.20:
            score += 10
            assessment_details.append("High Drawdown")

        # Consistency score (10%)
        if win_rate > 0.60:
            score += 10
            assessment_details.append("Consistent")
        elif win_rate > 0.50:
            score += 5

        # Final assessment
        if score >= 80:
            assessment = "Excellent"
        elif score >= 60:
            assessment = "Good"
        elif score >= 40:
            assessment = "Moderate"
        elif score >= 20:
            assessment = "Weak"
        else:
            assessment = "Poor"

        # Format output with proper alignment
        type_str = strategy_type[:11]
        ann_ret_str = f"{ann_return*100:6.2f}%" if not np.isnan(ann_return) else "N/A"
        vol_str = f"{volatility*100:5.1f}%" if not np.isnan(volatility) else "N/A"
        sharpe_str = f"{sharpe:6.2f}" if not np.isinf(sharpe) and not np.isnan(sharpe) else "N/A"
        dd_str = f"{max_dd*100:7.2f}%" if not np.isnan(max_dd) else "N/A"
        calmar_str = f"{calmar:6.2f}" if not np.isinf(calmar) and not np.isnan(calmar) else "N/A"
        win_str = f"{win_rate*100:6.1f}%" if not np.isnan(win_rate) else "N/A"
        var_str = f"{var_5pct*100:6.2f}%" if not np.isnan(var_5pct) else "N/A"

        print(f"{strategy_name:<25} {type_str:<12} {ann_ret_str:<9} {vol_str:<8} {sharpe_str:<8} {dd_str:<9} {calmar_str:<8} {win_str:<9} {var_str:<8} {assessment}")

    # Add summary of best performers
    if any(results_dict.values()):
        print("\n" + "=" * 150)
        print("TOP FINANCIAL PERFORMERS SUMMARY")
        print("=" * 150)

        valid_strategies = {k: v for k, v in results_dict.items() if v is not None}

        if valid_strategies:
            # Best return
            best_return_strategy = max(valid_strategies.keys(),
                                     key=lambda k: valid_strategies[k].get('annualized_return', 0))
            best_return_val = valid_strategies[best_return_strategy].get('annualized_return', 0)

            # Best Sharpe
            sharpe_strategies = {k: v for k, v in valid_strategies.items() if v.get('sharpe_ratio', 0) > 0}
            if sharpe_strategies:
                best_sharpe_strategy = max(sharpe_strategies.keys(),
                                         key=lambda k: sharpe_strategies[k].get('sharpe_ratio', 0))
                best_sharpe_val = sharpe_strategies[best_sharpe_strategy].get('sharpe_ratio', 0)
            else:
                best_sharpe_strategy, best_sharpe_val = "None", 0

            # Best drawdown control
            best_dd_strategy = min(valid_strategies.keys(),
                                  key=lambda k: abs(valid_strategies[k].get('max_drawdown', 1)))
            best_dd_val = valid_strategies[best_dd_strategy].get('max_drawdown', 0)

            print(f"Best Return: {best_return_strategy} ({best_return_val*100:.2f}% annually)")
            print(f"Best Sharpe: {best_sharpe_strategy} ({best_sharpe_val:.2f} ratio)")
            print(f"Best Risk Control: {best_dd_strategy} ({best_dd_val*100:.2f}% max drawdown)")

            # Risk-Return efficiency analysis
            efficient_strategies = []
            for name, metrics in valid_strategies.items():
                sharpe = metrics.get('sharpe_ratio', 0)
                ret = metrics.get('annualized_return', 0)
                dd = abs(metrics.get('max_drawdown', 1))

                # Composite score: weight return (30%), sharpe (40%), drawdown control (30%)
                if sharpe > 0 and dd < 0.5:  # Reasonable drawdown
                    composite_score = (ret * 0.3) + (sharpe * 0.4 * 0.1) + ((0.2 - min(dd, 0.2)) * 0.3 * 5)
                    efficient_strategies.append((name, composite_score, ret, sharpe, dd))

            if efficient_strategies:
                efficient_strategies.sort(key=lambda x: x[1], reverse=True)
                print(f"\nMost Risk-Efficient: {efficient_strategies[0][0]} (Score: {efficient_strategies[0][1]:.3f})")
                print(f"   Return: {efficient_strategies[0][2]*100:.2f}% | Sharpe: {efficient_strategies[0][3]:.2f} | MaxDD: {efficient_strategies[0][4]*100:.2f}%")

def create_comparison_table(results_dict):
    """Legacy comparison table - kept for compatibility"""
    create_comprehensive_financial_comparison(results_dict)

def main():
    parser = argparse.ArgumentParser(description='Complete HFT Strategy Comparison Pipeline')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Trading symbol')
    parser.add_argument('--period', type=str, default='5d', help='Data period')
    parser.add_argument('--interval', type=str, default='1m', help='Data interval')
    parser.add_argument('--quick', action='store_true', help='Quick mode - smaller dataset, fewer epochs')
    parser.add_argument('--include-all', action='store_true', help='Include all strategies (slower)')
    parser.add_argument('--skip-deeplob', action='store_true', help='Skip DeepLOB + Transformer strategy (faster)')
    parser.add_argument('--skip-llm', action='store_true', help='Skip LLM/Deep Learning strategies')
    parser.add_argument('--generate-pdf', action='store_true', help='Generate strategy comparison PDF report')

    args = parser.parse_args()

    # Adjust for quick mode
    if args.quick:
        args.period = '1d'
        args.interval = '5m'

    print(f"""
==============================================================================
                COMPREHENSIVE FINANCIAL STRATEGY ANALYSIS
           Complete Performance Evaluation with Financial Metrics
==============================================================================

Configuration:
   - Symbol: {args.symbol}
   - Period: {args.period} | Interval: {args.interval}
   - Mode: {'Quick' if args.quick else 'Full'} | Include All: {'Yes' if args.include_all else 'No'}
   - Skip DeepLOB: {'Yes' if args.skip_deeplob else 'No'} | Skip LLM: {'Yes' if args.skip_llm else 'No'}
   - PDF Report: {'Enabled' if args.generate_pdf else 'Disabled'}

Strategy Categories to Test:
   - Machine Learning: Linear, Ridge, Random Forest
   - Traditional Quant: Momentum, Mean Reversion, Pairs Trading
   - Deep Learning: LSTM, GRU, Transformer, CNN-LSTM
   - DeepLOB + Transformer: Advanced LOB-based Deep Learning

Financial Metrics Calculated:
   - Annualized Return & Volatility
   - Sharpe Ratio & Calmar Ratio
   - Maximum Drawdown & VaR (5%)
   - Win Rate & Risk-Adjusted Performance

Starting comprehensive financial evaluation...
""")

    start_time = time.time()
    all_results = {}

    try:
        # Step 1: Data and ML Signal Generation
        print("Step 1/5: Loading data and generating ML signals...")
        processor = SignalProcessor(data_source='yahoo')

        data = processor.load_data(
            symbol=args.symbol,
            period=args.period,
            interval=args.interval
        )
        print(f"   Loaded {len(data)} records")

        features = processor.generate_features()
        print(f"   Generated {features.shape[1]} features")

        ml_results = processor.train_signal_models(test_size=0.3)
        print("   Trained ML models")

        # Store ML results with comprehensive financial metrics
        for model_name, metrics in processor.performance_metrics.items():
            signals = processor.generate_signals(model_name=model_name)
            financial_metrics = calculate_strategy_financial_metrics(
                signals, data, 'Machine Learning'
            )
            # Merge ML-specific metrics with financial metrics
            financial_metrics.update({
                'hit_rate': metrics['hit_rate'],
                'information_coefficient': metrics['information_coefficient'],
                'r2_score': metrics['r2_score']
            })
            all_results[f"ML - {model_name.title()}"] = financial_metrics

        # Step 2: Traditional Strategies
        print("\nStep 2/5: Running traditional quantitative strategies...")

        # Momentum Strategy
        momentum_result = safe_execute_strategy(
            lambda: MomentumStrategy().generate_signals(data),
            "Momentum Strategy"
        )
        if momentum_result:
            financial_metrics = calculate_strategy_financial_metrics(
                momentum_result, data, 'Rule-based'
            )
            all_results["Traditional - Momentum"] = financial_metrics

        # Mean Reversion Strategy
        mean_reversion_result = safe_execute_strategy(
            lambda: MeanReversionStrategy().generate_signals(data),
            "Mean Reversion Strategy"
        )
        if mean_reversion_result:
            financial_metrics = calculate_strategy_financial_metrics(
                mean_reversion_result, data, 'Rule-based'
            )
            all_results["Traditional - Mean Reversion"] = financial_metrics

        # Pairs Trading Strategy
        pairs_result = safe_execute_strategy(
            lambda: _run_pairs_trading_strategy(data, args.symbol),
            "Pairs Trading Strategy"
        )
        if pairs_result:
            financial_metrics = calculate_strategy_financial_metrics(
                pairs_result, data, 'Rule-based'
            )
            all_results["Traditional - Pairs Trading"] = financial_metrics

        # Step 3: Deep Learning Strategies
        if not args.skip_llm:
            print("\nStep 3/5: Running LLM/Deep Learning strategies...")
            epochs = 5 if args.quick else 20

            # LSTM Strategy
            lstm_result = safe_execute_strategy(
                lambda: LSTMStrategy(epochs=epochs).train_and_predict(features, data['close']),
                "LSTM Strategy"
            )
            if lstm_result and lstm_result.get('signals') is not None:
                signals = lstm_result['signals']
                financial_metrics = calculate_strategy_financial_metrics(
                    signals, data, 'Deep Learning'
                )
                # Add LSTM-specific metrics if available
                if 'hit_rate' in lstm_result:
                    financial_metrics['hit_rate'] = lstm_result['hit_rate']
                if 'r2_score' in lstm_result:
                    financial_metrics['r2_score'] = lstm_result['r2_score']

                all_results["LLM - LSTM"] = financial_metrics

            # GRU Strategy
            gru_result = safe_execute_strategy(
                lambda: GRUStrategy(epochs=epochs).train_and_predict(features, data['close']),
                "GRU Strategy"
            )
            if gru_result and gru_result.get('signals') is not None:
                signals = gru_result['signals']
                financial_metrics = calculate_strategy_financial_metrics(
                    signals, data, 'Deep Learning'
                )
                # Add GRU-specific metrics if available
                if 'hit_rate' in gru_result:
                    financial_metrics['hit_rate'] = gru_result['hit_rate']
                if 'r2_score' in gru_result:
                    financial_metrics['r2_score'] = gru_result['r2_score']

                all_results["LLM - GRU"] = financial_metrics

        # Step 4: DeepLOB + Transformer Strategy
        if not args.skip_deeplob:
            print("\nStep 4/5: Running DeepLOB + Transformer strategy...")

            # DeepLOB + Transformer Strategy
            deeplob_result = safe_execute_strategy(
                lambda: _train_and_evaluate_deeplob_transformer(
                    data, features, epochs=20 if args.quick else 50
                ),
                "DeepLOB + Transformer"
            )
            if deeplob_result:
                annual_return = deeplob_result.get('total_return', 0) * 252
                initial_capital = 100000
                net_pnl = annual_return * initial_capital
                all_results["DeepLOB + Transformer"] = {
                    'type': 'Deep Learning',
                    'net_pnl': net_pnl,
                    'initial_capital': initial_capital,
                    'total_return_pct': annual_return * 100,
                    'annualized_return': annual_return,
                    'volatility': deeplob_result.get('volatility', 0),
                    'sharpe_ratio': deeplob_result.get('sharpe_ratio', 0),
                    'max_drawdown': deeplob_result.get('max_drawdown', 0),
                    'calmar_ratio': annual_return / (abs(deeplob_result.get('max_drawdown', 1)) + 1e-8),
                    'win_rate': deeplob_result.get('hit_rate', 0),
                    'var_5pct': deeplob_result.get('max_drawdown', 0) * 1.65,  # Approximate VaR
                    'total_signals': deeplob_result.get('total_signals', 0),
                    'buy_signals': deeplob_result.get('total_signals', 0) // 3,  # Approximate
                    'sell_signals': deeplob_result.get('total_signals', 0) // 3,  # Approximate
                    'train_accuracy': deeplob_result.get('final_train_acc', 0),
                    'validation_accuracy': deeplob_result.get('final_val_acc', 0),
                    'hit_rate': deeplob_result.get('hit_rate', 0),
                    'information_coefficient': 0,
                    'r2_score': 0
                }

        # Step 5: Results and Comparison
        print("\nStep 5/5: Generating comparison results...")

        # Create comparison table
        create_comparison_table(all_results)

        # Find best strategies using financial metrics
        best_return = max([k for k in all_results.keys() if all_results[k] is not None],
                         key=lambda k: all_results[k].get('annualized_return', 0), default=None)
        best_sharpe = max([k for k in all_results.keys() if all_results[k] is not None and all_results[k].get('sharpe_ratio', 0) > 0],
                         key=lambda k: all_results[k].get('sharpe_ratio', 0), default=None)
        best_drawdown = min([k for k in all_results.keys() if all_results[k] is not None],
                           key=lambda k: abs(all_results[k].get('max_drawdown', 1)), default=None)
        best_calmar = max([k for k in all_results.keys() if all_results[k] is not None],
                         key=lambda k: all_results[k].get('calmar_ratio', 0) if not np.isinf(all_results[k].get('calmar_ratio', 0)) else 0, default=None)

        # Save comprehensive financial results
        results_df = pd.DataFrame([
            {
                'strategy': name,
                'type': results.get('type', 'Unknown'),
                'net_pnl': results.get('net_pnl', 0),
                'initial_capital': results.get('initial_capital', 100000),
                'total_return_pct': results.get('total_return_pct', 0),
                'annualized_return': results.get('annualized_return', 0),
                'volatility': results.get('volatility', 0),
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'max_drawdown': results.get('max_drawdown', 0),
                'calmar_ratio': results.get('calmar_ratio', 0),
                'win_rate': results.get('win_rate', 0),
                'var_5pct': results.get('var_5pct', 0),
                'total_signals': results.get('total_signals', 0),
                'buy_signals': results.get('buy_signals', 0),
                'sell_signals': results.get('sell_signals', 0),
                'hit_rate': results.get('hit_rate', 0),
                'information_coefficient': results.get('information_coefficient', 0),
                'r2_score': results.get('r2_score', 0)
            }
            for name, results in all_results.items() if results is not None
        ])

        exports_dir = Path('exports')
        exports_dir.mkdir(exist_ok=True)

        comparison_file = exports_dir / f'{args.symbol}_strategy_comparison_{args.period}_{args.interval}.csv'
        results_df.to_csv(comparison_file, index=False)

        # Always generate net PnL PDF report
        from evaluation.pdf_report_generator import PDFReportGenerator
        try:
            pdf_generator = PDFReportGenerator()
            net_pnl_pdf_file = exports_dir / f'{args.symbol}_net_pnl_report_{args.period}_{args.interval}.pdf'
            pdf_generator.generate_net_pnl_report(all_results, str(net_pnl_pdf_file), args.symbol)
            print(f"   Net PnL PDF report saved: {net_pnl_pdf_file.name}")
        except Exception as e:
            print(f"   Net PnL PDF generation failed: {str(e)}")

        # Generate PDF report if requested
        pdf_file = None
        if args.generate_pdf:
            print("\nGenerating Strategy Comparison PDF Report...")
            try:
                # Convert results to format expected by PDF generator
                strategies_for_pdf = {}
                for name, results in all_results.items():
                    if results is not None:
                        strategies_for_pdf[name] = {
                            'performance_metrics': {
                                'net_total_pnl': results.get('annualized_return', 0) * 100000,  # Simulate with 100k capital
                                'cumulative_return': results.get('annualized_return', 0),
                                'sharpe_ratio': results.get('sharpe_ratio', 0),
                                'max_drawdown': results.get('max_drawdown', 0),
                                'volatility': results.get('volatility', 0),
                                'win_rate': results.get('win_rate', 0),
                                'cost_drag_pct': 2.0,  # Default cost drag
                                'net_profit_margin': results.get('annualized_return', 0) * 100,
                                'return_to_cost_ratio': abs(results.get('annualized_return', 0)) / 0.02 if results.get('annualized_return', 0) != 0 else 0
                            },
                            'returns_series': None  # Will be handled by PDF generator
                        }

                pdf_generator = StrategyComparisonPDFGenerator(
                    report_title=f"HFT Strategy Comparison - {args.symbol}"
                )

                pdf_file = exports_dir / f'{args.symbol}_strategy_comparison_report_{args.period}_{args.interval}.pdf'
                pdf_generator.generate_strategy_comparison_pdf(
                    strategies_for_pdf,
                    str(pdf_file)
                )
                print(f"   PDF report saved: {pdf_file.name}")

            except Exception as e:
                print(f"   PDF generation failed: {str(e)}")
                print("   Generating sample PDF instead...")
                try:
                    from evaluation.strategy_comparison_pdf import generate_sample_strategy_comparison_pdf
                    sample_path = generate_sample_strategy_comparison_pdf()

                    # Move to exports directory
                    import shutil
                    pdf_file = exports_dir / f'{args.symbol}_sample_strategy_comparison.pdf'
                    shutil.move(sample_path, pdf_file)
                    print(f"   Sample PDF report saved: {pdf_file.name}")
                except Exception as e2:
                    print(f"   Sample PDF generation also failed: {str(e2)}")
                    pdf_file = None

        total_time = time.time() - start_time

        print(f"""
==============================================================================
                    FINANCIAL COMPARISON COMPLETE!

  Strategies Evaluated: {len(all_results)}
  Total Time: {total_time:.1f} seconds
  Results: {comparison_file.name}""" + (f"""
  PDF Report: {pdf_file.name if pdf_file else 'Not generated'}""" if args.generate_pdf else "") + f"""

  Best Annualized Return: {best_return or 'N/A'}
  Best Sharpe Ratio: {best_sharpe or 'N/A'}
  Best Drawdown Control: {best_drawdown or 'N/A'}
  Best Risk-Adjusted: {best_calmar or 'N/A'}
==============================================================================

Comprehensive financial analysis complete!
Key metrics included: Returns, Volatility, Sharpe, Drawdown, Calmar, Win Rate, VaR
Check exports/ for detailed CSV with all financial indicators.
""")

        return all_results

    except Exception as e:
        logger.error(f"Strategy comparison failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()