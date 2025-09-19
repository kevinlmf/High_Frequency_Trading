"""
Mean reversionstrategy
Based onstrategy
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from scipy import stats

logger = logging.getLogger(__name__)


class MeanReversionStrategy:
    """
Mean reversionstrategyImplementation
"""

    def __init__(self,
                 lookback_window: int = 20,
                 zscore_threshold: float = 2.0,
                 half_life: int = 10,
                 min_periods: int = 10):
        """
InitializeMean reversionstrategy

        Args:
            lookback_window: Calculate
            zscore_threshold: Z-score，signals
            half_life: Mean reversion（For）
            min_periods: Calculate
"""
        self.lookback_window = lookback_window
        self.zscore_threshold = zscore_threshold
        self.half_life = half_life
        self.min_periods = min_periods

        self.name = "Mean Reversion Strategy"
        self.is_fitted = False

        # 
        self.price_mean = None
        self.price_std = None
        self.correlation_lookback = 50

        logger.info(f"Mean Reversion Strategy initialized:")
        logger.info(f"  - lookback_window: {lookback_window}")
        logger.info(f"  - zscore_threshold: {zscore_threshold}")
        logger.info(f"  - half_life: {half_life}")

    def calculate_rolling_stats(self, prices: pd.Series) -> tuple:
        """
Calculate
"""
        # 
        sma = prices.rolling(self.lookback_window, min_periods=self.min_periods).mean()
        rolling_std = prices.rolling(self.lookback_window, min_periods=self.min_periods).std()

        # 
        alpha = 1 - np.exp(-np.log(2) / self.half_life)
        ewma = prices.ewm(alpha=alpha, min_periods=self.min_periods).mean()
        ewm_std = prices.ewm(alpha=alpha, min_periods=self.min_periods).std()

        return sma, rolling_std, ewma, ewm_std

    def calculate_zscore(self, prices: pd.Series, mean: pd.Series, std: pd.Series) -> pd.Series:
        """
CalculateZ-score
"""
        zscore = (prices - mean) / (std + 1e-8)  # 
        return zscore

    def detect_regime(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        """
（vs）
"""
        # Using
        def rolling_regression_slope(y, window=20):
            slopes = []
            for i in range(len(y)):
                start_idx = max(0, i - window + 1)
                end_idx = i + 1
                if end_idx - start_idx >= self.min_periods:
                    x = np.arange(end_idx - start_idx)
                    y_window = y[start_idx:end_idx].values
                    if len(y_window) > 1 and not np.all(np.isnan(y_window)):
                        slope, _, r_value, _, _ = stats.linregress(x, y_window)
                        # R²
                        trend_strength = slope * (r_value ** 2)
                        slopes.append(trend_strength)
                    else:
                        slopes.append(0)
                else:
                    slopes.append(0)
            return pd.Series(slopes, index=y.index)

        # Calculate
        trend_strength = rolling_regression_slope(prices, self.lookback_window)

        # Calculate
        returns = prices.pct_change()
        rolling_vol = returns.rolling(self.lookback_window).std()
        vol_percentile = rolling_vol.rolling(self.correlation_lookback).rank(pct=True)

        # ：-1（，Mean reversion）, 0（，Mean reversion）, 1（）
        regime = pd.Series(0, index=prices.index)
        regime[np.abs(trend_strength) > np.abs(trend_strength).rolling(50).quantile(0.8)] = -1  # 
        regime[vol_percentile > 0.8] = 1  # 

        return regime

    def calculate_bollinger_bands(self, prices: pd.Series) -> tuple:
        """
Calculate
"""
        sma = prices.rolling(self.lookback_window).mean()
        std = prices.rolling(self.lookback_window).std()

        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)

        #  (0-1，0.5)
        bb_position = (prices - lower_band) / (upper_band - lower_band + 1e-8)

        return upper_band, lower_band, bb_position

    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """
strategyparameters

        Args:
            data: OHLCVdata
"""
        logger.info("Fitting mean reversion strategy...")

        # Validatedata
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        prices = data['close']
        returns = prices.pct_change().dropna()

        # CalculateForparametersOptimize
        historical_vol = returns.std()

        # AdjustZ-score：Using
        vol_adjustment = min(2.0, max(0.5, historical_vol / 0.02))
        self.adaptive_zscore_threshold = self.zscore_threshold * vol_adjustment

        # TestMean reversion
        adf_statistics = []
        for window in [10, 15, 20, 25, 30]:
            try:
                from statsmodels.tsa.stattools import adfuller
                rolling_zscore = self.calculate_zscore(
                    prices,
                    prices.rolling(window).mean(),
                    prices.rolling(window).std()
                ).dropna()

                if len(rolling_zscore) > 30:  # Hasdata
                    adf_stat, p_value, _, _, _, _ = adfuller(rolling_zscore)
                    adf_statistics.append((window, adf_stat, p_value))
            except:
                continue

        # （ADF，Mean reversion）
        if adf_statistics:
            best_window = min(adf_statistics, key=lambda x: x[1])[0]
            self.adaptive_lookback = best_window
        else:
            self.adaptive_lookback = self.lookback_window

        # Calculate
        # Using
        autocorr_lags = []
        for lag in range(1, min(50, len(returns) // 4)):
            if lag < len(returns):
                autocorr = returns.autocorr(lag)
                if not np.isnan(autocorr):
                    autocorr_lags.append((lag, autocorr))

        if autocorr_lags:
            # 0.5lag
            half_life_estimate = next((lag for lag, corr in autocorr_lags if corr < 0.5), self.half_life)
            self.adaptive_half_life = min(half_life_estimate, self.half_life + 10)
        else:
            self.adaptive_half_life = self.half_life

        logger.info(f"Strategy fitted with adaptive parameters:")
        logger.info(f"  - adaptive_zscore_threshold: {self.adaptive_zscore_threshold:.3f}")
        logger.info(f"  - adaptive_lookback: {self.adaptive_lookback}")
        logger.info(f"  - adaptive_half_life: {self.adaptive_half_life}")

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

        logger.info("Generating mean reversion signals...")

        prices = data['close']
        volume = data['volume']

        # 1. Calculate
        sma, rolling_std, ewma, ewm_std = self.calculate_rolling_stats(prices)

        # 2. CalculateZ-scores (Using)
        zscore_sma = self.calculate_zscore(prices, sma, rolling_std)
        zscore_ewma = self.calculate_zscore(prices, ewma, ewm_std)

        # 3. Calculate
        upper_band, lower_band, bb_position = self.calculate_bollinger_bands(prices)

        # 4. 
        market_regime = self.detect_regime(prices, volume)

        # 5. CalculateRSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        rsi_deviation = np.abs(rsi - 50) / 50  # RSI

        # 6. Signal strengthCalculate
        # Mean reversionsignals：Z-score，signals
        base_signal_sma = -np.tanh(zscore_sma / self.adaptive_zscore_threshold)  # 
        base_signal_ewma = -np.tanh(zscore_ewma / self.adaptive_zscore_threshold)

        # signals：signals
        bb_signal = np.where(bb_position > 0.8, -1,  # ，
                            np.where(bb_position < 0.2, 1, 0))  # ，
        bb_signal = bb_signal * (1 - np.abs(bb_position - 0.5) * 2)  # signals

        # RSIMean reversionsignals
        rsi_signal = np.where(rsi > 70, -rsi_deviation,  # 
                             np.where(rsi < 30, rsi_deviation, 0))  # 

        # Signal strength
        signal_strength = (
            0.4 * base_signal_sma +      # SMA Z-score40%
            0.3 * base_signal_ewma +     # EWMA Z-score30%
            0.2 * bb_signal +            # 20%
            0.1 * rsi_signal             # RSI10%
        )

        # 7. 
        # InSignal strength
        regime_filter = np.where(market_regime == -1, 0.3,  # signals
                                np.where(market_regime == 1, 1.2, 1.0))  # signals
        signal_strength = signal_strength * regime_filter

        # 8. GenerateDiscrete signals
        signals = pd.Series(0, index=data.index)
        signals[signal_strength > self.adaptive_zscore_threshold / 2] = 1   # （）
        signals[signal_strength < -self.adaptive_zscore_threshold / 2] = -1  # （）

        # 9. Calculate
        # Based onZ-score
        confidence = np.abs(signal_strength) / (self.adaptive_zscore_threshold + 1e-8)
        confidence = confidence.clip(0, 1)

        # Adjust
        confidence = confidence * np.where(market_regime == -1, 0.5,  # 
                                          np.where(market_regime == 0, 1.2, 1.0))  # 
        confidence = confidence.clip(0, 1)

        # 10. signals
        # signals
        signals[confidence < 0.4] = 0

        # ：
        signals = self._apply_continuity_filter(signals, min_hold_periods=3)

        logger.info(f"Generated {len(signals)} mean reversion signals")
        logger.info(f"Signal distribution: Buy={sum(signals==1)}, Sell={sum(signals==-1)}, Hold={sum(signals==0)}")

        return {
            'signals': signals,
            'signal_strength': signal_strength,
            'confidence': confidence,
            'metadata': {
                'strategy': 'Mean Reversion',
                'zscore_sma': zscore_sma,
                'zscore_ewma': zscore_ewma,
                'bollinger_position': bb_position,
                'market_regime': market_regime,
                'rsi': rsi,
                'sma': sma,
                'ewma': ewma,
                'upper_band': upper_band,
                'lower_band': lower_band
            }
        }

    def _apply_continuity_filter(self, signals: pd.Series, min_hold_periods: int = 3) -> pd.Series:
        """
Application，
"""
        filtered_signals = signals.copy()
        current_position = 0
        hold_counter = 0

        for i in range(len(signals)):
            new_signal = signals.iloc[i]

            if new_signal != current_position:
                if hold_counter < min_hold_periods and current_position != 0:
                    # IfHasTime，When
                    filtered_signals.iloc[i] = current_position
                else:
                    # 
                    current_position = new_signal
                    hold_counter = 0
            else:
                hold_counter += 1

        return filtered_signals

    def get_strategy_info(self) -> Dict[str, Any]:
        """
Getstrategy
"""
        return {
            'name': self.name,
            'type': 'Traditional - Mean Reversion',
            'parameters': {
                'lookback_window': self.lookback_window,
                'zscore_threshold': self.zscore_threshold,
                'half_life': self.half_life,
                'adaptive_zscore_threshold': getattr(self, 'adaptive_zscore_threshold', None),
                'adaptive_lookback': getattr(self, 'adaptive_lookback', None),
                'adaptive_half_life': getattr(self, 'adaptive_half_life', None)
            },
            'is_fitted': self.is_fitted
        }


if __name__ == "__main__":
    # TestMean reversionstrategy
    from ...signal_engine.data_sources.yahoo_finance import YahooFinanceSource

    # GetTestdata
    source = YahooFinanceSource()
    data = source.download_data("AAPL", period="5d", interval="1m")

    # CreateRunMean reversionstrategy
    mean_reversion = MeanReversionStrategy(lookback_window=20, zscore_threshold=1.5)

    # Generatesignals
    results = mean_reversion.generate_signals(data)

    print("\n=== Mean Reversion Strategy Results ===")
    print(f"Strategy Info: {mean_reversion.get_strategy_info()}")
    print(f"Signal Summary:")
    print(f"  Buy signals: {sum(results['signals'] == 1)}")
    print(f"  Sell signals: {sum(results['signals'] == -1)}")
    print(f"  Hold signals: {sum(results['signals'] == 0)}")
    print(f"  Mean confidence: {results['confidence'].mean():.3f}")
    print(f"  Mean |signal strength|: {results['signal_strength'].abs().mean():.3f}")

    # 
    metadata = results['metadata']
    print(f"  Mean Z-score (SMA): {metadata['zscore_sma'].abs().mean():.3f}")
    print(f"  Mean Z-score (EWMA): {metadata['zscore_ewma'].abs().mean():.3f}")
    print(f"  Market regime distribution: {metadata['market_regime'].value_counts().to_dict()}")