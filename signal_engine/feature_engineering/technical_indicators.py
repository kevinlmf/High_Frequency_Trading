"""
Technical Indicators Calculation Module
Based on original HFT_Signal 48+ technical features engineering
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Technical indicators calculator - Implement 48+ technical indicators
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize technical indicators calculator

        Args:
            data: DataFrame containing OHLCV data
        """
        self.data = data.copy()
        self.features = pd.DataFrame()

        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def calculate_all_features(self) -> pd.DataFrame:
        """
        Calculate all 48+ technical features
        """
        logger.info("Calculating all technical indicators...")

        # Basic price features
        self._calculate_price_features()

        # Return features
        self._calculate_return_features()

        # Moving average features
        self._calculate_moving_average_features()

        # Momentum indicators
        self._calculate_momentum_features()

        # Volatility features
        self._calculate_volatility_features()

        # Volume features
        self._calculate_volume_features()

        # Technical indicators
        self._calculate_technical_indicators()

        # Microstructure features
        self._calculate_microstructure_features()

        # Time-based features
        self._calculate_time_features()

        logger.info(f"Generated {len(self.features.columns)} technical features")
        return self.features

    def _calculate_price_features(self):
        """
        Price-related features
        """
        data = self.data

        # Basic price levels
        self.features['close_price'] = data['close']
        self.features['high_price'] = data['high']
        self.features['low_price'] = data['low']
        self.features['open_price'] = data['open']

        # Price spreads
        self.features['hl_spread'] = data['high'] - data['low']
        self.features['oc_spread'] = data['open'] - data['close']

        # Price ratios
        self.features['close_to_high'] = data['close'] / data['high']
        self.features['close_to_low'] = data['close'] / data['low']
        self.features['hl_ratio'] = data['high'] / data['low']

    def _calculate_return_features(self):
        """Return features"""
        data = self.data

        # Simple returns
        self.features['return_1'] = data['close'].pct_change(1)
        self.features['return_5'] = data['close'].pct_change(5)
        self.features['return_10'] = data['close'].pct_change(10)

        # Log returns
        self.features['log_return_1'] = np.log(data['close'] / data['close'].shift(1))

        # Cumulative returns
        self.features['cum_return_10'] = (data['close'] / data['close'].shift(10) - 1)

    def _calculate_moving_average_features(self):
        """Moving average features"""
        close = self.data['close']

        # Simple moving averages
        for window in [5, 10, 20, 50]:
            ma_col = f'ma_{window}'
            self.features[ma_col] = close.rolling(window).mean()
            self.features[f'price_to_{ma_col}'] = close / self.features[ma_col]
            self.features[f'{ma_col}_slope'] = self.features[ma_col].diff(3)

        # Exponential moving averages
        for span in [5, 10, 20]:
            ema_col = f'ema_{span}'
            self.features[ema_col] = close.ewm(span=span).mean()

    def _calculate_momentum_features(self):
        """Momentum indicators"""
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        self.features['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = close.ewm(12).mean()
        ema_26 = close.ewm(26).mean()
        self.features['macd'] = ema_12 - ema_26
        self.features['macd_signal'] = self.features['macd'].ewm(9).mean()
        self.features['macd_histogram'] = self.features['macd'] - self.features['macd_signal']

        # Stochastic oscillator
        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        self.features['stoch_k'] = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        self.features['stoch_d'] = self.features['stoch_k'].rolling(3).mean()

    def _calculate_volatility_features(self):
        """Volatility features"""
        data = self.data
        close = data['close']

        # Rolling volatility
        returns = close.pct_change()
        for window in [5, 10, 20]:
            self.features[f'volatility_{window}'] = returns.rolling(window).std() * np.sqrt(252)

        # ATR (Average True Range)
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        self.features['atr'] = true_range.rolling(14).mean()

        # Bollinger Bands
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        self.features['bollinger_upper'] = ma20 + (2 * std20)
        self.features['bollinger_lower'] = ma20 - (2 * std20)
        self.features['bollinger_position'] = (close - self.features['bollinger_lower']) / (self.features['bollinger_upper'] - self.features['bollinger_lower'])

    def _calculate_volume_features(self):
        """Volume features"""
        volume = self.data['volume']
        close = self.data['close']

        # Volume moving average
        self.features['volume_ma_10'] = volume.rolling(10).mean()
        self.features['volume_ratio'] = volume / self.features['volume_ma_10']

        # Price-volume metrics
        self.features['price_volume'] = close * volume

        # VWAP (Volume Weighted Average Price)
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        self.features['vwap'] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        self.features['price_to_vwap'] = close / self.features['vwap']

        # Money Flow Index
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
        self.features['mfi'] = 100 - (100 / (1 + (positive_flow / negative_flow)))

    def _calculate_technical_indicators(self):
        """Additional technical indicators"""
        close = self.data['close']

        # Williams %R indicator
        high_14 = self.data['high'].rolling(14).max()
        low_14 = self.data['low'].rolling(14).min()
        self.features['williams_r'] = -100 * (high_14 - close) / (high_14 - low_14)

        # CCI (Commodity Channel Index)
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        self.features['cci'] = (typical_price - sma_tp) / (0.015 * mad)

    def _calculate_microstructure_features(self):
        """Market microstructure features (simulated)"""
        data = self.data

        # Spread features (approximated using high-low as proxy for bid-ask spread)
        self.features['spread'] = data['high'] - data['low']
        self.features['spread_ma'] = self.features['spread'].rolling(10).mean()
        self.features['relative_spread'] = self.features['spread'] / data['close']

        # Price impact (approximated)
        self.features['price_impact'] = np.abs(data['close'] - data['open']) / data['volume'].replace(0, np.nan)

    def _calculate_time_features(self):
        """Time-based features"""
        if 'timestamp' in self.data.columns:
            timestamps = pd.to_datetime(self.data['timestamp'])
            self.features['hour'] = timestamps.dt.hour
            self.features['minute'] = timestamps.dt.minute
            self.features['day_of_week'] = timestamps.dt.dayofweek

        # Time cyclical features
        n_periods = len(self.data)
        self.features['time_sin'] = np.sin(2 * np.pi * np.arange(n_periods) / (24 * 60))  # Assumes 1-minute data
        self.features['time_cos'] = np.cos(2 * np.pi * np.arange(n_periods) / (24 * 60))

    def get_feature_summary(self) -> Dict[str, Any]:
        """Get feature summary statistics"""
        return {
            'total_features': len(self.features.columns),
            'feature_names': list(self.features.columns),
            'data_shape': self.features.shape,
            'null_counts': self.features.isnull().sum().to_dict(),
            'feature_stats': self.features.describe().to_dict()
        }


if __name__ == "__main__":
    # Test technical indicators
    from ..data_sources.yahoo_finance import YahooFinanceSource

    source = YahooFinanceSource()
    data = source.download_data("AAPL", period="5d", interval="1m")

    indicators = TechnicalIndicators(data)
    features = indicators.calculate_all_features()

    print(f"Generated {len(features.columns)} features")
    print("Feature columns:", list(features.columns)[:10])  # Show first 10 features