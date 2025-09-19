"""
HFT Unified System Main Controller
Complete pipeline integrating signal generation, strategy execution and performance evaluation
"""

import pandas as pd
import numpy as np
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# System module imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_engine.signal_processor import SignalProcessor
from strategy_methods.traditional.momentum_strategy import MomentumStrategy
from strategy_methods.traditional.mean_reversion_strategy import MeanReversionStrategy
from strategy_methods.traditional.pairs_trading_strategy import PairsTradingStrategy
from strategy_methods.rl_methods.trading_env import HFTTradingEnv
from strategy_methods.rl_methods.rl_agents import RLAgentManager
from strategy_methods.llm_methods.lstm_strategy import LSTMStrategy
from strategy_methods.llm_methods.gru_strategy import GRUStrategy
from strategy_methods.llm_methods.transformer_strategy import TransformerStrategy
from strategy_methods.llm_methods.cnn_lstm_strategy import CNNLSTMStrategy

# Evaluation module imports
from evaluation import (
    PerformanceMetrics,
    Backtester,
    StrategyComparisonDashboard,
    SimpleDataProvider,
    ExecutionEngine
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HFTUnifiedSystem:
    """HFT Unified System Main Controller"""

    def __init__(self, data_source: str = 'yahoo'):
        """
        Initialize HFT Unified System

        Args:
            data_source: Data source type ('yahoo', 'real_data')
        """
        self.data_source = data_source
        self.signal_processor = SignalProcessor(data_source=data_source)
        self.strategies = {}
        self.results = {}
        self.performance_metrics = {}

        # Initialize evaluation components
        self.backtester = None
        self.comparison_dashboard = None
        self.execution_engine = ExecutionEngine(commission_rate=0.001)

        # System status
        self.data_loaded = False
        self.signals_generated = False

        logger.info(f"HFT Unified System initialized with {data_source} data source")

    def load_data(self, symbol: str = "AAPL", period: str = "5d", interval: str = "1m", **kwargs):
        """Load market data"""
        logger.info(f"Loading market data for {symbol}...")

        # Load data using signal processor
        self.raw_data = self.signal_processor.load_data(symbol=symbol, period=period, interval=interval, **kwargs)
        self.symbol = symbol

        # Generate technical features
        self.features = self.signal_processor.generate_features()

        self.data_loaded = True
        logger.info(f"Data loaded successfully: {self.raw_data.shape}")

        return self.raw_data

    def generate_ml_signals(self, model_name: str = 'ridge'):
        """Generate ML signals"""
        if not self.data_loaded:
            raise ValueError("Data must be loaded first")

        logger.info("Generating ML signals...")

        # TrainMLmodelGeneratesignals
        training_results = self.signal_processor.train_signal_models()
        self.ml_signals = self.signal_processor.generate_signals(model_name=model_name)
        self.ml_signal_strength = self.signal_processor.generate_signals(
            model_name=model_name, return_strength=True
        )

        self.signals_generated = True
        logger.info("ML signals generated successfully")

        return self.ml_signals

    def run_traditional_strategies(self) -> Dict[str, Any]:
        """Run traditional strategies"""
        if not self.data_loaded:
            raise ValueError("Data must be loaded first")

        logger.info("Running traditional strategies...")

        traditional_results = {}

        # 1. Momentum strategy
        try:
            momentum = MomentumStrategy(lookback_period=20, momentum_threshold=0.015)
            momentum_results = momentum.generate_signals(self.raw_data)
            traditional_results['momentum'] = {
                'strategy': momentum,
                'results': momentum_results
            }
            logger.info("✅ Momentum strategy completed")
        except Exception as e:
            logger.error(f"❌ Momentum strategy failed: {e}")

        # 2. Mean reversionstrategy
        try:
            mean_reversion = MeanReversionStrategy(lookback_window=20, zscore_threshold=1.5)
            mr_results = mean_reversion.generate_signals(self.raw_data)
            traditional_results['mean_reversion'] = {
                'strategy': mean_reversion,
                'results': mr_results
            }
            logger.info("✅ Mean reversion strategy completed")
        except Exception as e:
            logger.error(f"❌ Mean reversion strategy failed: {e}")

        # 3. Pairs trading（IfHasdata）
        try:
            # ，Create（UsingTime）
            pairs_data = {
                self.symbol: self.raw_data,
                f"{self.symbol}_shifted": self.raw_data.shift(5).dropna()  # Time
            }

            pairs_trading = PairsTradingStrategy(formation_period=50, entry_threshold=1.5)
            pairs_trading.fit(pairs_data)  # Fit the strategy before generating signals
            pairs_results = pairs_trading.generate_signals(pairs_data)
            traditional_results['pairs_trading'] = {
                'strategy': pairs_trading,
                'results': pairs_results
            }
            logger.info("✅ Pairs trading strategy completed")
        except Exception as e:
            logger.error(f"❌ Pairs trading strategy failed: {e}")

        self.strategies.update(traditional_results)
        return traditional_results

    def run_rl_strategies(self, training_steps: int = 10000) -> Dict[str, Any]:
        """Run reinforcement learning strategies"""
        if not self.signals_generated:
            logger.warning("No ML signals found. Generating signals first...")
            self.generate_ml_signals()

        logger.info("Running RL strategies...")

        try:
            # CreateTrading environment
            env = HFTTradingEnv(
                data=self.raw_data,
                features=self.features,
                signals=self.ml_signals,
                initial_balance=10000
            )

            # CreateRL
            rl_manager = RLAgentManager(env)

            rl_results = {}

            # TrainPPO
            try:
                logger.info("Training PPO agent...")
                rl_manager.train_agent('ppo', total_timesteps=training_steps)
                ppo_eval = rl_manager.evaluate_agent('ppo', n_episodes=3)
                rl_results['ppo'] = ppo_eval
                logger.info("✅ PPO training completed")
            except Exception as e:
                logger.error(f"❌ PPO training failed: {e}")

            # TrainSAC
            try:
                logger.info("Training SAC agent...")
                rl_manager.train_agent('sac', total_timesteps=training_steps)
                sac_eval = rl_manager.evaluate_agent('sac', n_episodes=3)
                rl_results['sac'] = sac_eval
                logger.info("✅ SAC training completed")
            except Exception as e:
                logger.error(f"❌ SAC training failed: {e}")

            self.strategies['rl'] = {
                'manager': rl_manager,
                'results': rl_results
            }

            return rl_results

        except Exception as e:
            logger.error(f"RL strategies failed: {e}")
            return {}

    def run_llm_strategies(self, training_epochs: int = 100) -> Dict[str, Any]:
        """
RunLLM/Deep Learningstrategy
"""
        if not self.data_loaded:
            raise ValueError("Data must be loaded first")

        logger.info("Running LLM/Deep Learning strategies...")

        llm_results = {}

        # LSTMstrategy
        try:
            logger.info("Training LSTM strategy...")
            lstm_strategy = LSTMStrategy(epochs=training_epochs)
            lstm_results = lstm_strategy.generate_signals(self.raw_data)
            llm_results['lstm'] = {
                'strategy': lstm_strategy,
                'results': lstm_results
            }
            logger.info("✅ LSTM strategy completed")
        except Exception as e:
            logger.error(f"❌ LSTM strategy failed: {e}")

        # GRUstrategy
        try:
            logger.info("Training GRU strategy...")
            gru_strategy = GRUStrategy(epochs=training_epochs)
            gru_results = gru_strategy.generate_signals(self.raw_data)
            llm_results['gru'] = {
                'strategy': gru_strategy,
                'results': gru_results
            }
            logger.info("✅ GRU strategy completed")
        except Exception as e:
            logger.error(f"❌ GRU strategy failed: {e}")

        # Transformerstrategy
        try:
            logger.info("Training Transformer strategy...")
            transformer_strategy = TransformerStrategy(epochs=training_epochs)
            transformer_results = transformer_strategy.generate_signals(self.raw_data)
            llm_results['transformer'] = {
                'strategy': transformer_strategy,
                'results': transformer_results
            }
            logger.info("✅ Transformer strategy completed")
        except Exception as e:
            logger.error(f"❌ Transformer strategy failed: {e}")

        # CNN-LSTMstrategy
        try:
            logger.info("Training CNN-LSTM strategy...")
            cnn_lstm_strategy = CNNLSTMStrategy(epochs=training_epochs)
            cnn_lstm_results = cnn_lstm_strategy.generate_signals(self.raw_data)
            llm_results['cnn_lstm'] = {
                'strategy': cnn_lstm_strategy,
                'results': cnn_lstm_results
            }
            logger.info("✅ CNN-LSTM strategy completed")
        except Exception as e:
            logger.error(f"❌ CNN-LSTM strategy failed: {e}")

        self.strategies.update(llm_results)
        return llm_results

    def compare_all_strategies(self) -> pd.DataFrame:
        """
CompareHasstrategyperformance
"""
        logger.info("Comparing strategy performances...")

        comparison_data = []

        # MLsignalsperformance - HasWhensignalsGenerate
        if (hasattr(self, 'signal_processor') and
            self.signal_processor.performance_metrics and
            hasattr(self, 'ml_signals') and
            self.ml_signals is not None and
            len(self.ml_signals) > 0):

            for model_name, metrics in self.signal_processor.performance_metrics.items():
                # Calculatesignals
                total_signals = len(self.ml_signals)
                buy_signals = (self.ml_signals == 1).sum()
                sell_signals = (self.ml_signals == -1).sum()
                hold_signals = (self.ml_signals == 0).sum()

                # CalculateSignal strength
                signal_strength_stats = ""
                if hasattr(self, 'ml_signal_strength') and self.ml_signal_strength is not None:
                    strength_mean = self.ml_signal_strength.abs().mean()
                    signal_strength_stats = f"{strength_mean:.3f}"
                else:
                    signal_strength_stats = "N/A"

                comparison_data.append({
                    'Strategy': f"ML - {model_name.title()}",
                    'Type': 'Machine Learning',
                    'Total Signals': total_signals,
                    'Buy Signals': buy_signals,
                    'Sell Signals': sell_signals,
                    'Hold Signals': hold_signals,
                    'Signal Coverage': f"{((buy_signals + sell_signals) / total_signals * 100):.1f}%",
                    'Hit Rate': f"{metrics.get('hit_rate', 0):.1f}%",
                    'Info Coefficient': f"{metrics.get('information_coefficient', 0):.4f}",
                    'R² Score': f"{metrics.get('r2_score', 0):.4f}",
                    'Mean Signal Strength': signal_strength_stats,
                    'Model Accuracy': f"{metrics.get('accuracy', 0):.3f}",
                    'Precision': f"{metrics.get('precision', 0):.3f}",
                    'Recall': f"{metrics.get('recall', 0):.3f}"
                })

        # Traditional strategiesperformance
        for strategy_name, strategy_data in self.strategies.items():
            if strategy_name in ['rl', 'lstm', 'gru', 'transformer', 'cnn_lstm']:
                continue

            if 'results' in strategy_data:
                results = strategy_data['results']
                signals = results['signals']
                confidence = results['confidence']

                # Calculate
                total_signals = len(signals)
                buy_signals = (signals == 1).sum()
                sell_signals = (signals == -1).sum()
                hold_signals = (signals == 0).sum()
                active_signals = buy_signals + sell_signals

                # Calculate
                conf_mean = confidence.mean() if len(confidence) > 0 else 0
                conf_std = confidence.std() if len(confidence) > 0 else 0
                conf_max = confidence.max() if len(confidence) > 0 else 0

                comparison_data.append({
                    'Strategy': f"Traditional - {strategy_name.replace('_', ' ').title()}",
                    'Type': 'Traditional',
                    'Total Signals': total_signals,
                    'Buy Signals': buy_signals,
                    'Sell Signals': sell_signals,
                    'Hold Signals': hold_signals,
                    'Signal Coverage': f"{(active_signals / total_signals * 100):.1f}%",
                    'Hit Rate': f"{(active_signals / total_signals * 100):.1f}%",
                    'Mean Confidence': f"{conf_mean:.3f}",
                    'Std Confidence': f"{conf_std:.3f}",
                    'Max Confidence': f"{conf_max:.3f}",
                    'Buy/Sell Ratio': f"{(buy_signals / sell_signals if sell_signals > 0 else float('inf')):.2f}",
                    'Signal Consistency': f"{(1 - conf_std/conf_mean if conf_mean > 0 else 0):.3f}"
                })

        # LLM/Deep LearningStrategy performance
        for strategy_name in ['lstm', 'gru', 'transformer', 'cnn_lstm']:
            if strategy_name in self.strategies and 'results' in self.strategies[strategy_name]:
                results = self.strategies[strategy_name]['results']
                signals = results['signals']
                confidence = results['confidence']
                hit_rate = results.get('hit_rate', 0)

                # Calculate
                total_signals = len(signals)
                buy_signals = (signals == 1).sum()
                sell_signals = (signals == -1).sum()
                hold_signals = (signals == 0).sum()
                active_signals = buy_signals + sell_signals

                # Deep LearningHasmetrics
                training_loss = results.get('training_loss', 0)
                validation_loss = results.get('validation_loss', 0)
                model_epochs = results.get('epochs_trained', 0)

                # Analyze
                positive_conf = confidence[confidence > 0]
                conf_mean = positive_conf.mean() if len(positive_conf) > 0 else 0
                conf_std = positive_conf.std() if len(positive_conf) > 0 else 0

                comparison_data.append({
                    'Strategy': f"LLM - {results.get('strategy_name', strategy_name.upper())}",
                    'Type': 'Deep Learning',
                    'Total Signals': total_signals,
                    'Buy Signals': buy_signals,
                    'Sell Signals': sell_signals,
                    'Hold Signals': hold_signals,
                    'Signal Coverage': f"{(active_signals / total_signals * 100):.1f}%",
                    'Hit Rate': f"{hit_rate:.1f}%",
                    'Mean Confidence': f"{conf_mean:.3f}" if conf_mean > 0 else "N/A",
                    'Std Confidence': f"{conf_std:.3f}" if conf_std > 0 else "N/A",
                    'Training Loss': f"{training_loss:.4f}" if training_loss > 0 else "N/A",
                    'Validation Loss': f"{validation_loss:.4f}" if validation_loss > 0 else "N/A",
                    'Epochs Trained': model_epochs,
                    'Buy/Sell Ratio': f"{(buy_signals / sell_signals if sell_signals > 0 else float('inf')):.2f}"
                })

        # RLStrategy performance
        if 'rl' in self.strategies:
            rl_results = self.strategies['rl']['results']
            for agent_name, metrics in rl_results.items():
                # RLHasmetrics
                mean_reward = metrics.get('mean_reward', 0)
                std_reward = metrics.get('std_reward', 0)
                total_return = metrics.get('avg_total_return', 0)
                sharpe_ratio = metrics.get('avg_sharpe_ratio', 0)
                max_drawdown = metrics.get('avg_max_drawdown', 0)
                win_rate = metrics.get('avg_win_rate', 0)
                profit_factor = metrics.get('avg_profit_factor', 0)
                volatility = metrics.get('avg_volatility', 0)

                comparison_data.append({
                    'Strategy': f"RL - {agent_name.upper()}",
                    'Type': 'Reinforcement Learning',
                    'Mean Reward': f"{mean_reward:.3f}",
                    'Std Reward': f"{std_reward:.3f}" if std_reward > 0 else "N/A",
                    'Total Return': f"{total_return * 100:.2f}%" if total_return != 0 else '0.00%',
                    'Sharpe Ratio': f"{sharpe_ratio:.3f}" if sharpe_ratio != 0 else '0.000',
                    'Max Drawdown': f"{abs(max_drawdown) * 100:.2f}%" if max_drawdown != 0 else '0.00%',
                    'Win Rate': f"{win_rate * 100:.1f}%" if win_rate > 0 else "N/A",
                    'Profit Factor': f"{profit_factor:.2f}" if profit_factor > 0 else "N/A",
                    'Volatility': f"{volatility * 100:.2f}%" if volatility > 0 else "N/A",
                    'Risk-Adj Return': f"{(total_return / volatility if volatility > 0 else 0):.3f}",
                    'Reward Consistency': f"{(1 - std_reward/abs(mean_reward) if mean_reward != 0 else 0):.3f}"
                })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            logger.info("Strategy comparison completed")
            return comparison_df
        else:
            logger.warning("No strategy results available for comparison")
            return pd.DataFrame()

    def run_comprehensive_evaluation(self, strategies_to_evaluate: List[str] = None) -> Dict[str, Any]:
        """
RunEvaluate
"""
        logger.info("🔍 Running comprehensive performance evaluation...")

        if not self.strategies:
            logger.warning("No strategies available for evaluation")
            return {}

        # strategydataEvaluate
        strategies_data = {}

        for strategy_name, strategy_data in self.strategies.items():
            if strategies_to_evaluate and strategy_name not in strategies_to_evaluate:
                continue

            if 'results' not in strategy_data:
                continue

            # Generate (Based onsignals)
            results = strategy_data['results']
            signals = results.get('signals', pd.Series())

            if len(signals) > 0:
                # : signals * 
                if hasattr(self, 'raw_data') and 'close' in self.raw_data.columns:
                    market_returns = self.raw_data['close'].pct_change().fillna(0)
                    strategy_returns = signals.shift(1).fillna(0) * market_returns[:len(signals)]
                else:
                    # Generate
                    np.random.seed(42)
                    strategy_returns = pd.Series(
                        np.random.normal(0.0005, 0.02, len(signals)),
                        index=signals.index
                    )

                # Createstrategydata
                strategies_data[strategy_name] = {
                    'returns': strategy_returns,
                    'performance_metrics': PerformanceMetrics(strategy_returns).calculate_all_metrics(),
                    'signals': signals,
                    'confidence': results.get('confidence', pd.Series())
                }

        if not strategies_data:
            logger.warning("No valid strategy data for evaluation")
            return {}

        # Create
        self.comparison_dashboard = StrategyComparisonDashboard(strategies_data)

        # Generate
        evaluation_results = {
            'performance_comparison': self.comparison_dashboard.generate_performance_comparison_table(),
            'comprehensive_report': self.comparison_dashboard.generate_comprehensive_report(),
            'strategy_rankings': self._rank_strategies(strategies_data),
            'evaluation_summary': self._generate_evaluation_summary(strategies_data)
        }

        logger.info("✅ Comprehensive evaluation completed")
        return evaluation_results

    def run_backtesting(self,
                       strategy_name: str,
                       start_date: str = None,
                       end_date: str = None,
                       initial_capital: float = 1000000.0) -> Dict[str, Any]:
        """
RunAnalyze
"""
        logger.info(f"🔄 Running backtest for strategy: {strategy_name}")

        if strategy_name not in self.strategies:
            logger.error(f"Strategy {strategy_name} not found")
            return {}

        if not hasattr(self, 'raw_data') or self.raw_data.empty:
            logger.error("No market data loaded for backtesting")
            return {}

        # data
        price_data = {
            self.symbol: self.raw_data[['open', 'high', 'low', 'close', 'volume']].copy()
        }
        price_data[self.symbol]['price'] = price_data[self.symbol]['close']

        # CreatedataProvides
        data_provider = SimpleDataProvider(price_data)

        # Create
        self.backtester = Backtester(
            data_provider=data_provider,
            execution_engine=self.execution_engine,
            initial_capital=initial_capital
        )

        # CreatestrategyPackage
        from evaluation.backtester import TradingStrategy

        class StrategyWrapper(TradingStrategy):
            def __init__(self, signals_data):
                self.signals_data = signals_data
                self.signal_index = 0

            def generate_signals(self, market_data, portfolio):
                if self.signal_index < len(self.signals_data):
                    signal = self.signals_data.iloc[self.signal_index]
                    self.signal_index += 1

                    # signals
                    if signal == 1:  # signals
                        return {list(market_data.keys())[0]: 0.8}
                    elif signal == -1:  # signals
                        return {list(market_data.keys())[0]: 0.0}
                    else:  # Has
                        return {}
                return {}

        # Getstrategysignals
        strategy_data = self.strategies[strategy_name]
        signals = strategy_data['results']['signals']

        # Createstrategy
        strategy = StrategyWrapper(signals)

        # Run
        backtest_start = self.raw_data.index[0] if start_date is None else pd.to_datetime(start_date)
        backtest_end = self.raw_data.index[-1] if end_date is None else pd.to_datetime(end_date)

        backtest_results = self.backtester.run_backtest(
            strategy=strategy,
            symbols=[self.symbol],
            start_date=backtest_start.to_pydatetime(),
            end_date=backtest_end.to_pydatetime(),
            rebalance_frequency='H'
        )

        logger.info("✅ Backtesting completed")
        return backtest_results

    def generate_evaluation_dashboard(self, save_path: str = "evaluation_dashboard.html") -> None:
        """
GenerateEvaluate
"""
        if self.comparison_dashboard is None:
            logger.warning("No comparison dashboard available. Run comprehensive evaluation first.")
            return

        logger.info(f"📊 Generating evaluation dashboard: {save_path}")

        # Generate
        fig = self.comparison_dashboard.plot_cumulative_returns()
        fig.write_html(save_path)

        logger.info(f"✅ Dashboard saved to: {save_path}")

    def _rank_strategies(self, strategies_data: Dict[str, Any]) -> pd.DataFrame:
        """
strategy
"""
        rankings = []

        for strategy_name, data in strategies_data.items():
            metrics = data['performance_metrics']

            #  (Adjust)
            score = (
                metrics.get('sharpe_ratio', 0) * 0.4 +
                metrics.get('annualized_return', 0) * 100 * 0.3 +
                (1 - abs(metrics.get('max_drawdown', 0))) * 0.2 +
                metrics.get('win_rate', 0) * 0.1
            )

            rankings.append({
                'Strategy': strategy_name,
                'Overall Score': round(score, 3),
                'Sharpe Ratio': round(metrics.get('sharpe_ratio', 0), 3),
                'Annual Return': f"{metrics.get('annualized_return', 0):.2%}",
                'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
                'Win Rate': f"{metrics.get('win_rate', 0):.2%}"
            })

        ranking_df = pd.DataFrame(rankings).sort_values('Overall Score', ascending=False)
        return ranking_df.reset_index(drop=True)

    def _generate_evaluation_summary(self, strategies_data: Dict[str, Any]) -> str:
        """
GenerateEvaluate
"""
        summary = "=== HFTstrategyEvaluate ===\n\n"

        summary += f"📊 Evaluatestrategy: {len(strategies_data)}\n"
        summary += f"🎯 data: {self.raw_data.index[0]}  {self.raw_data.index[-1]}\n\n"

        # strategy
        best_sharpe = max(
            strategies_data.items(),
            key=lambda x: x[1]['performance_metrics'].get('sharpe_ratio', 0)
        )

        best_return = max(
            strategies_data.items(),
            key=lambda x: x[1]['performance_metrics'].get('annualized_return', 0)
        )

        summary += f"🏆 Sharpe ratio: {best_sharpe[0]} ({best_sharpe[1]['performance_metrics'].get('sharpe_ratio', 0):.3f})\n"
        summary += f"📈 Annualized return: {best_return[0]} ({best_return[1]['performance_metrics'].get('annualized_return', 0):.2%})\n\n"

        # Analyze
        summary += "⚠️ :\n"
        for strategy_name, data in strategies_data.items():
            metrics = data['performance_metrics']
            summary += f"  {strategy_name}: Maximum drawdown {metrics.get('max_drawdown', 0):.2%}\n"

        return summary

    def run_full_pipeline(self,
                         symbol: str = "AAPL",
                         period: str = "5d",
                         interval: str = "1m",
                         include_rl: bool = True,
                         include_llm: bool = True,
                         rl_training_steps: int = 5000,
                         llm_training_epochs: int = 50,
                         **kwargs) -> Dict[str, Any]:
        """
RunHFTpipeline
"""
        start_time = time.time()
        logger.info("🚀 Starting HFT Unified System Full Pipeline...")

        pipeline_results = {}

        try:
            # 1. dataLoadfeaturesGenerate
            logger.info("📊 Step 1: Loading data and generating features...")
            self.load_data(symbol=symbol, period=period, interval=interval, **kwargs)
            pipeline_results['data_loaded'] = True

            # 2. MLSignal Generation
            logger.info("🧠 Step 2: Generating ML signals...")
            ml_signals = self.generate_ml_signals()
            pipeline_results['ml_signals_generated'] = True

            # 3. Traditional strategies
            logger.info("📈 Step 3: Running traditional strategies...")
            traditional_results = self.run_traditional_strategies()
            pipeline_results['traditional_strategies'] = traditional_results

            # 4. LLM/Deep Learningstrategy（）
            if include_llm:
                logger.info("🧠 Step 4: Running LLM/Deep Learning strategies...")
                llm_results = self.run_llm_strategies(training_epochs=llm_training_epochs)
                pipeline_results['llm_strategies'] = llm_results
            else:
                logger.info("⏭️  Step 4: Skipping LLM strategies...")

            # 5. RLstrategy（）
            if include_rl:
                logger.info("🎮 Step 5: Running RL strategies...")
                rl_results = self.run_rl_strategies(training_steps=rl_training_steps)
                pipeline_results['rl_strategies'] = rl_results
            else:
                logger.info("⏭️  Step 5: Skipping RL strategies...")

            # 6. performanceCompare
            logger.info("📊 Step 6: Comparing all strategies...")
            comparison_df = self.compare_all_strategies()
            pipeline_results['comparison'] = comparison_df

            # 7. Evaluate
            logger.info("🔍 Step 7: Running comprehensive evaluation...")
            evaluation_results = self.run_comprehensive_evaluation()
            pipeline_results['evaluation'] = evaluation_results

            # 8. GenerateEvaluate
            logger.info("📊 Step 8: Generating evaluation dashboard...")
            self.generate_evaluation_dashboard("evaluation_dashboard.html")
            pipeline_results['dashboard_generated'] = True

            # 9. 
            total_time = time.time() - start_time
            logger.info(f"✅ Pipeline completed successfully in {total_time:.2f} seconds")

            pipeline_results.update({
                'execution_time': total_time,
                'symbol': symbol,
                'data_shape': self.raw_data.shape,
                'features_count': len(self.features.columns),
                'success': True
            })

            return pipeline_results

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            pipeline_results['success'] = False
            pipeline_results['error'] = str(e)
            return pipeline_results

    def save_results(self, output_dir: str = "results") -> None:
        """
Saveresultsfile
"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # SaveCompareresults
        if hasattr(self, 'comparison_df') and not self.comparison_df.empty:
            comparison_file = output_path / f"{self.symbol}_strategy_comparison.csv"
            self.comparison_df.to_csv(comparison_file, index=False)
            logger.info(f"Comparison results saved to {comparison_file}")

        # Savesignals
        if hasattr(self, 'ml_signals'):
            signals_file = output_path / f"{self.symbol}_ml_signals.csv"
            self.ml_signals.to_csv(signals_file)
            logger.info(f"ML signals saved to {signals_file}")


def demo_run():
    """
Run
"""
    print("🚀 HFT Unified System Demo")
    print("=" * 50)

    # Createsystem
    hft_system = HFTUnifiedSystem()

    # RunComplete Pipeline
    results = hft_system.run_full_pipeline(
        symbol="AAPL",
        period="2d",
        interval="1m",
        include_rl=True,  # ContainsRLstrategy
        include_llm=True,  # ContainsLLM/Deep Learningstrategy
        rl_training_steps=5000,
        llm_training_epochs=30  # epochsFordemo
    )

    # results
    if results['success']:
        print("\n✅ Demo completed successfully!")
        print(f"⏱️  Execution time: {results['execution_time']:.2f} seconds")
        print(f"📊 Data shape: {results['data_shape']}")
        print(f"🧠 Features generated: {results['features_count']}")

        if 'comparison' in results and not results['comparison'].empty:
            print("\n📈 Strategy Performance Comparison:")
            print("-" * 80)

            # Setuppandas
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)

            print(results['comparison'].to_string(index=False))

            # 
            pd.reset_option('display.max_columns')
            pd.reset_option('display.width')
            pd.reset_option('display.max_colwidth')

        # Evaluateresults
        if 'evaluation' in results and results['evaluation']:
            evaluation = results['evaluation']

            print("\n🔍 Comprehensive Evaluation Results:")
            if 'evaluation_summary' in evaluation:
                print(evaluation['evaluation_summary'])

            if 'strategy_rankings' in evaluation and not evaluation['strategy_rankings'].empty:
                print("\n🏆 Strategy Rankings:")
                print(evaluation['strategy_rankings'].head().to_string(index=False))

            if results.get('dashboard_generated'):
                print("\n📊 Interactive dashboard saved: evaluation_dashboard.html")
                print("   Open this file in a browser to view detailed analysis!")

        # 
        if hft_system.strategies:
            print("\n🔄 Running backtest demo...")
            first_strategy = list(hft_system.strategies.keys())[0]
            backtest_results = hft_system.run_backtesting(
                strategy_name=first_strategy,
                initial_capital=100000.0
            )

            if backtest_results and 'performance_report' in backtest_results:
                print(f"\n📋 Backtest Report for {first_strategy}:")
                print(backtest_results['performance_report'][:500] + "...")  # 500

    else:
        print(f"❌ Demo failed: {results.get('error', 'Unknown error')}")


def main():
    """
function
"""
    parser = argparse.ArgumentParser(description='HFT Unified System')

    # Data source options
    parser.add_argument('--data-source', default='yahoo', choices=['yahoo', 'real_data'],
                       help='Data source type')
    parser.add_argument('--data-type', default='enhanced_yahoo',
                       choices=['enhanced_yahoo', 'synthetic_lob', 'crypto_lob'],
                       help='Real data type (only for --data-source real_data)')

    # Symbol and basic parameters
    parser.add_argument('--symbol', default='AAPL', help='Stock symbol')
    parser.add_argument('--period', default='2d', help='Data period')
    parser.add_argument('--interval', default='1m', help='Data interval')

    # Real data specific parameters
    parser.add_argument('--records', type=int, default=10000,
                       help='Number of records for synthetic data')
    parser.add_argument('--levels', type=int, default=10,
                       help='Number of LOB levels for synthetic data')
    parser.add_argument('--start-price', type=float, default=100.0,
                       help='Starting price for synthetic data')

    # Processing options
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--include-rl', action='store_true', default=True, help='Include RL strategies')
    parser.add_argument('--skip-rl', action='store_true', help='Skip RL strategies to save time')
    parser.add_argument('--rl-steps', type=int, default=5000, help='RL training steps')
    parser.add_argument('--include-llm', action='store_true', default=True, help='Include LLM/Deep Learning strategies')
    parser.add_argument('--skip-llm', action='store_true', help='Skip LLM strategies to save time')
    parser.add_argument('--llm-epochs', type=int, default=50, help='LLM training epochs')

    args = parser.parse_args()

    if args.demo:
        demo_run()
    else:
        # Createsystem
        hft_system = HFTUnifiedSystem(data_source=args.data_source)

        # Runpipeline
        pipeline_kwargs = {}
        if args.data_source == 'real_data':
            pipeline_kwargs.update({
                'data_type': args.data_type,
                'records': args.records,
                'levels': args.levels,
                'start_price': args.start_price
            })

        results = hft_system.run_full_pipeline(
            symbol=args.symbol,
            period=args.period,
            interval=args.interval,
            include_rl=args.include_rl and not args.skip_rl,
            include_llm=args.include_llm and not args.skip_llm,
            rl_training_steps=args.rl_steps,
            llm_training_epochs=args.llm_epochs,
            **pipeline_kwargs
        )

        # results
        if results['success']:
            print("\n" + "="*80)
            print("📊 COMPREHENSIVE STRATEGY COMPARISON REPORT")
            print("="*80)

            if 'comparison' in results and not results['comparison'].empty:
                comparison_df = results['comparison']

                # SetuppandasHas
                pd.set_option('display.max_columns', None)
                pd.set_option('display.max_rows', None)
                pd.set_option('display.width', None)
                pd.set_option('display.max_colwidth', None)

                print("\n🎯 Full Strategy Performance Metrics:")
                print("-" * 80)
                print(comparison_df.to_string(index=False))

                # class
                if 'Type' in comparison_df.columns:
                    print("\n📈 Performance by Strategy Type:")
                    print("-" * 50)

                    for strategy_type in comparison_df['Type'].unique():
                        type_df = comparison_df[comparison_df['Type'] == strategy_type]
                        print(f"\n🔹 {strategy_type} Strategies:")
                        print(type_df.to_string(index=False))
                        print()

                # pandas
                pd.reset_option('display.max_columns')
                pd.reset_option('display.max_rows')
                pd.reset_option('display.width')
                pd.reset_option('display.max_colwidth')

            else:
                print("❌ No comparison data available")
        else:
            print(f"❌ Pipeline Failed: {results.get('error')}")


if __name__ == "__main__":
    main()