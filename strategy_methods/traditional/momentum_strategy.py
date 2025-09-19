"""
Momentum strategy
Based onstrategy
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MomentumStrategy:
    """
Momentum strategyImplementation
"""

    def __init__(self,
                 lookback_period: int = 20,
                 volume_lookback: int = 10,
                 momentum_threshold: float = 0.02):
        """
InitializeMomentum strategy

        Args:
            lookback_period: Calculate
            volume_lookback: 
            momentum_threshold: signals
"""
        self.lookback_period = lookback_period
        self.volume_lookback = volume_lookback
        self.momentum_threshold = momentum_threshold

        self.name = "Momentum Strategy"
        self.is_fitted = False

        logger.info(f"Momentum Strategy initialized with parameters:")
        logger.info(f"  - lookback_period: {lookback_period}")
        logger.info(f"  - volume_lookback: {volume_lookback}")
        logger.info(f"  - momentum_threshold: {momentum_threshold}")

    def calculate_price_momentum(self, prices: pd.Series) -> pd.Series:
        """
Calculate
"""
        # ：WhenN
        price_momentum = prices.pct_change(self.lookback_period)

        # ：
        short_momentum = prices.pct_change(5)  # 5
        long_momentum = prices.pct_change(self.lookback_period)  # 

        #  ()
        combined_momentum = 0.7 * short_momentum + 0.3 * long_momentum

        return combined_momentum

    def calculate_volume_momentum(self, volume: pd.Series) -> pd.Series:
        """
Calculate
"""
        # 
        vol_ma = volume.rolling(self.volume_lookback).mean()
        volume_ratio = volume / vol_ma

        # ：
        volume_momentum = volume.pct_change(self.volume_lookback)

        # metrics
        combined_vol_momentum = 0.6 * volume_ratio + 0.4 * volume_momentum

        return combined_vol_momentum

    def calculate_trend_strength(self, prices: pd.Series) -> pd.Series:
        """
Calculate
"""
        # Using
        ma5 = prices.rolling(5).mean()
        ma10 = prices.rolling(10).mean()
        ma20 = prices.rolling(20).mean()

        # ：
        trend_strength = (ma5 - ma20) / ma20

        # ：HasIsNot
        trend_consistency = ((ma5 > ma10) & (ma10 > ma20)).astype(int) * 2 - 1  # =1，=-1

        return trend_strength * trend_consistency

    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """
strategyparameters

        Args:
            data: OHLCVdata
"""
        logger.info("Fitting momentum strategy...")

        # Validatedata
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        # CalculateForparametersOptimize
        prices = data['close']
        returns = prices.pct_change()

        # AdjustBased on
        volatility = returns.rolling(30).std()
        mean_volatility = volatility.mean()

        # According toAdjust
        self.adaptive_threshold = self.momentum_threshold * (mean_volatility / 0.02)

        # Calculate（Based on）
        autocorrs = [returns.shift(i).corr(returns) for i in range(5, 50, 5)]
        optimal_lookback = (np.argmax(np.abs(autocorrs)) + 1) * 5

        if optimal_lookback > 0:
            self.adaptive_lookback = min(optimal_lookback, self.lookback_period + 10)
        else:
            self.adaptive_lookback = self.lookback_period

        logger.info(f"Strategy fitted with adaptive parameters:")
        logger.info(f"  - adaptive_threshold: {self.adaptive_threshold:.4f}")
        logger.info(f"  - adaptive_lookback: {self.adaptive_lookback}")

        self.is_fitted = True

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
Generate trading signals

        Args:
            data: OHLCVdata

        Returns:
            Containssignals、Signal strength
"""
        if not self.is_fitted:
            self.fit(data)

        logger.info("Generating momentum signals...")

        prices = data['close']
        volume = data['volume']

        # 1. CalculateMomentum indicators
        price_momentum = self.calculate_price_momentum(prices)
        volume_momentum = self.calculate_volume_momentum(volume)
        trend_strength = self.calculate_trend_strength(prices)

        # 2. CalculateMACDsignals
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()
        macd_histogram = macd - macd_signal

        # 3. CalculateRSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        rsi_momentum = (rsi - 50) / 50  # RSIMomentum indicators

        # 4. Signal strength
        signal_strength = (
            0.4 * price_momentum / self.adaptive_threshold +  # 40%
            0.2 * np.tanh(volume_momentum) +                  # 20%
            0.2 * trend_strength +                            # 20%
            0.1 * np.tanh(macd_histogram / prices.std()) +    # MACD10%
            0.1 * rsi_momentum                                # RSI10%
        )

        # 5. GenerateDiscrete signals
        signals = pd.Series(0, index=data.index)
        signals[signal_strength > self.adaptive_threshold] = 1   # 
        signals[signal_strength < -self.adaptive_threshold] = -1  # 

        # 6. Calculate
        # Based onSignal strengthmetrics
        abs_strength = signal_strength.abs()
        max_strength = abs_strength.rolling(50).max()
        confidence = abs_strength / (max_strength + 1e-8)

        # metrics
        momentum_consistency = np.abs([
            np.sign(price_momentum),
            np.sign(volume_momentum),
            np.sign(trend_strength),
            np.sign(macd_histogram),
            np.sign(rsi_momentum)
        ]).mean(axis=0)

        confidence = confidence * momentum_consistency
        confidence = confidence.clip(0, 1)

        # 7. signals
        # signals
        signals[confidence < 0.3] = 0

        # signals：
        signals = self._smooth_signals(signals, window=3)

        logger.info(f"Generated {len(signals)} momentum signals")
        logger.info(f"Signal distribution: Buy={sum(signals==1)}, Sell={sum(signals==-1)}, Hold={sum(signals==0)}")

        return {
            'signals': signals,
            'signal_strength': signal_strength,
            'confidence': confidence,
            'metadata': {
                'strategy': 'Momentum',
                'price_momentum': price_momentum,
                'volume_momentum': volume_momentum,
                'trend_strength': trend_strength,
                'macd': macd_histogram,
                'rsi_momentum': rsi_momentum
            }
        }

    def _smooth_signals(self, signals: pd.Series, window: int = 3) -> pd.Series:
        """
signals，
"""
        # Using
        smoothed = signals.copy()

        for i in range(window, len(signals)):
            window_signals = signals[i-window:i]
            if len(window_signals.unique()) == 1:
                # Ifsignals，
                continue
            else:
                # Using
                vote = window_signals.value_counts()
                if len(vote) > 0:
                    majority_signal = vote.idxmax()
                    smoothed.iloc[i] = majority_signal

        return smoothed

    def get_strategy_info(self) -> Dict[str, Any]:
        """
Getstrategy
"""
        return {
            'name': self.name,
            'type': 'Traditional - Momentum',
            'parameters': {
                'lookback_period': self.lookback_period,
                'volume_lookback': self.volume_lookback,
                'momentum_threshold': self.momentum_threshold,
                'adaptive_threshold': getattr(self, 'adaptive_threshold', None),
                'adaptive_lookback': getattr(self, 'adaptive_lookback', None)
            },
            'is_fitted': self.is_fitted
        }


if __name__ == "__main__":
    # TestMomentum strategy
    from ...signal_engine.data_sources.yahoo_finance import YahooFinanceSource

    # GetTestdata
    source = YahooFinanceSource()
    data = source.download_data("AAPL", period="5d", interval="1m")

    # CreateRunMomentum strategy
    momentum = MomentumStrategy(lookback_period=20, momentum_threshold=0.01)

    # Generatesignals
    results = momentum.generate_signals(data)

    print("\n=== Momentum Strategy Results ===")
    print(f"Strategy Info: {momentum.get_strategy_info()}")
    print(f"Signal Summary:")
    print(f"  Buy signals: {sum(results['signals'] == 1)}")
    print(f"  Sell signals: {sum(results['signals'] == -1)}")
    print(f"  Hold signals: {sum(results['signals'] == 0)}")
    print(f"  Mean confidence: {results['confidence'].mean():.3f}")
    print(f"  Mean signal strength: {results['signal_strength'].abs().mean():.3f}")