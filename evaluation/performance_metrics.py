"""
Performance Metrics Module for HFT Trading System
ContainsmetricsCalculatefunction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

class PerformanceMetrics:
    """
metricsCalculateclass
"""

    def __init__(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None,
                 risk_free_rate: float = 0.02):
        """
InitializemetricsCalculate

        Args:
            returns: strategy
            benchmark_returns: 
            risk_free_rate: No ()
"""
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.trading_days = 252

    def calculate_all_metrics(self) -> Dict[str, float]:
        """
CalculateHasmetrics
"""
        metrics = {}

        # metrics
        return_metrics = self._calculate_return_metrics()
        metrics.update(return_metrics)

        # metrics
        metrics.update(self._calculate_risk_metrics(return_metrics))

        # metrics
        metrics.update(self._calculate_trading_metrics())

        # metrics
        metrics.update(self._calculate_hft_metrics())

        return metrics

    def _calculate_return_metrics(self) -> Dict[str, float]:
        """
Calculatemetrics
"""
        metrics = {}

        # 
        cumulative_return = (1 + self.returns).prod() - 1
        metrics['cumulative_return'] = cumulative_return

        # Annualized return
        if len(self.returns) > 0:
            periods = len(self.returns) / self.trading_days
            if periods > 0 and (1 + cumulative_return) > 0:
                try:
                    annualized_return = (1 + cumulative_return) ** (1 / periods) - 1
                    metrics['annualized_return'] = annualized_return
                except (ZeroDivisionError, OverflowError, ValueError):
                    metrics['annualized_return'] = 0.0
            else:
                metrics['annualized_return'] = 0.0
        else:
            metrics['annualized_return'] = 0.0

        # 
        metrics['average_daily_return'] = self.returns.mean()

        #  ()
        if self.benchmark_returns is not None:
            excess_returns = self.returns - self.benchmark_returns
            metrics['excess_return'] = excess_returns.mean()
            metrics['annualized_excess_return'] = excess_returns.mean() * self.trading_days

        return metrics

    def _calculate_risk_metrics(self, return_metrics: Dict[str, float] = None) -> Dict[str, float]:
        """
Calculatemetrics
"""
        metrics = {}

        # 
        volatility = self.returns.std() * np.sqrt(self.trading_days)
        metrics['volatility'] = volatility

        # Sharpe ratio
        excess_return = self.returns.mean() * self.trading_days - self.risk_free_rate
        if volatility > 0:
            metrics['sharpe_ratio'] = excess_return / volatility
        else:
            metrics['sharpe_ratio'] = 0.0

        # Maximum drawdown
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        metrics['max_drawdown'] = max_drawdown

        # 
        if max_drawdown != 0 and return_metrics:
            annualized_return = return_metrics.get('annualized_return', 0)
            metrics['calmar_ratio'] = annualized_return / abs(max_drawdown)
        else:
            metrics['calmar_ratio'] = np.inf if max_drawdown != 0 else 0.0

        # VaR (5% )
        metrics['var_5pct'] = np.percentile(self.returns, 5)

        # CVaR ()
        var_threshold = metrics['var_5pct']
        cvar_returns = self.returns[self.returns <= var_threshold]
        metrics['cvar_5pct'] = cvar_returns.mean() if len(cvar_returns) > 0 else 0.0

        # Information ratio ()
        if self.benchmark_returns is not None:
            excess_returns = self.returns - self.benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(self.trading_days)
            if tracking_error > 0:
                metrics['information_ratio'] = (excess_returns.mean() * self.trading_days) / tracking_error
            else:
                metrics['information_ratio'] = 0.0

        # 
        metrics['skewness'] = self.returns.skew()
        metrics['kurtosis'] = self.returns.kurtosis()

        return metrics

    def _calculate_trading_metrics(self) -> Dict[str, float]:
        """
Calculatemetrics
"""
        metrics = {}

        #  (Win rate)
        positive_returns = self.returns > 0
        metrics['win_rate'] = positive_returns.mean()

        # Profit/Loss ratio
        positive_avg = self.returns[positive_returns].mean() if positive_returns.sum() > 0 else 0
        negative_avg = abs(self.returns[~positive_returns].mean()) if (~positive_returns).sum() > 0 else 0

        if negative_avg > 0:
            metrics['profit_loss_ratio'] = positive_avg / negative_avg
        else:
            metrics['profit_loss_ratio'] = np.inf if positive_avg > 0 else 0

        #  ()
        metrics['trade_frequency'] = len(self.returns)

        # /
        returns_sign = np.sign(self.returns)
        changes = np.diff(np.concatenate(([0], returns_sign, [0])))

        # 
        if len(changes) > 0:
            consecutive_wins = self._get_max_consecutive(returns_sign, 1)
            consecutive_losses = self._get_max_consecutive(returns_sign, -1)
            metrics['max_consecutive_wins'] = consecutive_wins
            metrics['max_consecutive_losses'] = consecutive_losses
        else:
            metrics['max_consecutive_wins'] = 0
            metrics['max_consecutive_losses'] = 0

        return metrics

    def _calculate_hft_metrics(self) -> Dict[str, float]:
        """
Calculatemetrics
"""
        metrics = {}

        # Time ()
        metrics['avg_holding_period'] = 1.0  # 

        #  (/)
        if self.returns.mean() != 0:
            metrics['return_stability'] = self.returns.std() / abs(self.returns.mean())
        else:
            metrics['return_stability'] = np.inf

        # 
        metrics['intraday_volatility'] = self.returns.std()

        #  (CheckIsNotInMean reversion)
        if len(self.returns) > 1:
            metrics['return_autocorr'] = self.returns.autocorr(lag=1)
        else:
            metrics['return_autocorr'] = 0.0

        # /
        metrics['max_daily_gain'] = self.returns.max()
        metrics['max_daily_loss'] = self.returns.min()

        # 
        metrics['tail_ratio'] = (abs(np.percentile(self.returns, 95)) /
                               abs(np.percentile(self.returns, 5))) if np.percentile(self.returns, 5) != 0 else np.inf

        return metrics

    def _get_max_consecutive(self, series: np.ndarray, value: int) -> int:
        """
Calculate
"""
        if len(series) == 0:
            return 0

        max_count = 0
        current_count = 0

        for val in series:
            if val == value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def calculate_rolling_metrics(self, window: int = 30) -> pd.DataFrame:
        """
Calculatemetrics
"""
        if len(self.returns) < window:
            warnings.warn(f"data {len(self.returns)}  {window}")
            return pd.DataFrame()

        rolling_metrics = pd.DataFrame(index=self.returns.index[window-1:])

        # Sharpe ratio
        rolling_returns = self.returns.rolling(window)
        rolling_metrics['rolling_sharpe'] = (
            (rolling_returns.mean() * self.trading_days - self.risk_free_rate) /
            (rolling_returns.std() * np.sqrt(self.trading_days))
        )

        # Maximum drawdown
        cumulative = (1 + self.returns).cumprod()
        rolling_max = cumulative.rolling(window).max()
        rolling_metrics['rolling_drawdown'] = (cumulative - rolling_max) / rolling_max

        # 
        rolling_metrics['rolling_volatility'] = (
            rolling_returns.std() * np.sqrt(self.trading_days)
        )

        # Win rate
        rolling_metrics['rolling_win_rate'] = (
            (self.returns > 0).rolling(window).mean()
        )

        return rolling_metrics

    def generate_performance_report(self) -> str:
        """
Generate
"""
        metrics = self.calculate_all_metrics()

        report = "=== strategy ===\n\n"

        # metrics
        report += "📈 metrics:\n"
        report += f"  : {metrics['cumulative_return']:.2%}\n"
        report += f"  Annualized return: {metrics['annualized_return']:.2%}\n"
        report += f"  : {metrics['average_daily_return']:.4%}\n"

        if 'excess_return' in metrics:
            report += f"  : {metrics['excess_return']:.4%}\n"

        report += "\n"

        # metrics
        report += "⚠️ metrics:\n"
        report += f"  : {metrics['volatility']:.2%}\n"
        report += f"  Maximum drawdown: {metrics['max_drawdown']:.2%}\n"
        report += f"  Sharpe ratio: {metrics['sharpe_ratio']:.3f}\n"
        report += f"  : {metrics['calmar_ratio']:.3f}\n"
        report += f"  VaR (5%): {metrics['var_5pct']:.3%}\n"
        report += f"  CVaR (5%): {metrics['cvar_5pct']:.3%}\n"

        if 'information_ratio' in metrics:
            report += f"  Information ratio: {metrics['information_ratio']:.3f}\n"

        report += "\n"

        # 
        report += "📊 :\n"
        report += f"  Win rate: {metrics['win_rate']:.2%}\n"
        report += f"  Profit/Loss ratio: {metrics['profit_loss_ratio']:.3f}\n"
        report += f"  : {metrics['trade_frequency']}\n"
        report += f"  : {metrics['max_consecutive_wins']} \n"
        report += f"  : {metrics['max_consecutive_losses']} \n"

        report += "\n"

        # 
        report += "⚡ :\n"
        report += f"  : {metrics['return_stability']:.3f}\n"
        report += f"  : {metrics['return_autocorr']:.3f}\n"
        report += f"  : {metrics['max_daily_gain']:.3%}\n"
        report += f"  : {metrics['max_daily_loss']:.3%}\n"
        report += f"  : {metrics['tail_ratio']:.3f}\n"

        return report


def calculate_strategy_comparison(strategies_returns: Dict[str, pd.Series],
                                benchmark_returns: Optional[pd.Series] = None) -> pd.DataFrame:
    """
Comparestrategymetrics

    Args:
        strategies_returns: strategy {strategy: }
        benchmark_returns: 

    Returns:
        ContainsHasstrategymetricsDataFrame
"""
    comparison_results = {}

    for strategy_name, returns in strategies_returns.items():
        metrics_calc = PerformanceMetrics(returns, benchmark_returns)
        metrics = metrics_calc.calculate_all_metrics()
        comparison_results[strategy_name] = metrics

    return pd.DataFrame(comparison_results).T


# ExampleUsing
if __name__ == "__main__":
    # GenerateExampledata
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=250, freq='D')

    # strategy
    strategy_returns = pd.Series(
        np.random.normal(0.001, 0.02, 250),
        index=dates
    )

    # 
    benchmark_returns = pd.Series(
        np.random.normal(0.0005, 0.015, 250),
        index=dates
    )

    # Calculatemetrics
    perf_metrics = PerformanceMetrics(strategy_returns, benchmark_returns)

    # Output
    print(perf_metrics.generate_performance_report())

    # Calculatemetrics
    rolling_metrics = perf_metrics.calculate_rolling_metrics(30)
    print("\nmetrics (5):")
    print(rolling_metrics.tail())