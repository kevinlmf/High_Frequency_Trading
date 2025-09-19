"""
HFT Signal Engine

Signal Generation，IntegrationData acquisition、Feature engineeringMLSignal Generation
"""

from .signal_processor import SignalProcessor
from .data_sources.yahoo_finance import YahooFinanceSource
from .feature_engineering.technical_indicators import TechnicalIndicators
from .ml_signals.signal_generator import MLSignalGenerator

__version__ = "1.0.0"

__all__ = [
    "SignalProcessor",
    "YahooFinanceSource",
    "TechnicalIndicators",
    "MLSignalGenerator"
]