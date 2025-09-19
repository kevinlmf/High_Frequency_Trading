"""
Evaluation Module for HFT Trading System
systemEvaluateModule

Contains:
- metricsCalculate (performance_metrics)
-  (backtester)
- strategy (comparison_dashboard)
"""

from .performance_metrics import PerformanceMetrics, calculate_strategy_comparison
from .backtester import (
    Backtester, TradingStrategy, DataProvider, SimpleDataProvider,
    ExecutionEngine, Trade, Position, PortfolioSnapshot,
    SimpleMovingAverageStrategy
)
from .comparison_dashboard import StrategyComparisonDashboard

__all__ = [
    # metrics
    'PerformanceMetrics',
    'calculate_strategy_comparison',

    # 
    'Backtester',
    'TradingStrategy',
    'DataProvider',
    'SimpleDataProvider',
    'ExecutionEngine',
    'Trade',
    'Position',
    'PortfolioSnapshot',
    'SimpleMovingAverageStrategy',

    # 
    'StrategyComparisonDashboard'
]

__version__ = "1.0.0"
__author__ = "HFT System Team"