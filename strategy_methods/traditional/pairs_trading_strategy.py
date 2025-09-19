"""
Pairs tradingstrategy
Based onstrategy
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class PairsTradingStrategy:
    """
Pairs tradingstrategyImplementation
"""

    def __init__(self,
                 formation_period: int = 60,
                 trading_period: int = 30,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 stop_loss_threshold: float = 3.0):
        """
InitializePairs tradingstrategy

        Args:
            formation_period: （For）
            trading_period: （Pairs trading）
            entry_threshold: Z-score
            exit_threshold: Z-score
            stop_loss_threshold: 
"""
        self.formation_period = formation_period
        self.trading_period = trading_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold

        self.name = "Pairs Trading Strategy"
        self.is_fitted = False

        # parameters
        self.pairs = []
        self.hedge_ratios = {}
        self.spread_means = {}
        self.spread_stds = {}

        logger.info(f"Pairs Trading Strategy initialized:")
        logger.info(f"  - formation_period: {formation_period}")
        logger.info(f"  - trading_period: {trading_period}")
        logger.info(f"  - entry_threshold: {entry_threshold}")
        logger.info(f"  - exit_threshold: {exit_threshold}")

    def find_cointegrated_pairs(self, price_data: Dict[str, pd.Series],
                               min_correlation: float = 0.8) -> List[Tuple[str, str, float]]:
        """


        Args:
            price_data: data {symbol: price_series}
            min_correlation: 

        Returns:
             [(symbol1, symbol2, cointegration_score)]
"""
        symbols = list(price_data.keys())
        pairs = []

        # CalculateHas
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]

                # GetTimedata
                common_index = price_data[symbol1].index.intersection(price_data[symbol2].index)
                if len(common_index) < self.formation_period:
                    continue

                price1 = price_data[symbol1].loc[common_index]
                price2 = price_data[symbol2].loc[common_index]

                # Check
                correlation = price1.corr(price2)
                if correlation < min_correlation:
                    continue

                try:
                    # 
                    cointegration_score = self._test_cointegration(price1, price2)
                    if cointegration_score < 0.05:  # p-value < 0.05In
                        pairs.append((symbol1, symbol2, cointegration_score))
                except Exception as e:
                    logger.warning(f"Cointegration test failed for {symbol1}-{symbol2}: {e}")
                    continue

        # 
        pairs.sort(key=lambda x: x[2])
        logger.info(f"Found {len(pairs)} cointegrated pairs")

        return pairs

    def _test_cointegration(self, price1: pd.Series, price2: pd.Series) -> float:
        """


        Args:
            price1, price2: 

        Returns:
            p-value
"""
        try:
            from statsmodels.tsa.stattools import coint
            # Engle-Granger
            coint_stat, p_value, critical_values = coint(price1, price2)
            return p_value
        except ImportError:
            # ：
            hedge_ratio = self._calculate_hedge_ratio(price1, price2)
            spread = price1 - hedge_ratio * price2

            # UsingADF
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_stat, p_value, _, _, _, _ = adfuller(spread.dropna())
                return p_value
            except ImportError:
                # Version：Using
                returns = spread.pct_change().dropna()
                if len(returns) < 30:
                    return 1.0
                # 1IfIs
                variance_ratio = returns.var() / (returns.shift(1).var() + 1e-8)
                # pseudo p-value
                return min(variance_ratio, 1.0 / variance_ratio)

    def _calculate_hedge_ratio(self, price1: pd.Series, price2: pd.Series) -> float:
        """
Calculate（Through）

        Args:
            price1, price2: 

        Returns:
            
"""
        # price1 = alpha + beta * price2 + error
        # hedge_ratio = beta
        reg = LinearRegression()
        reg.fit(price2.values.reshape(-1, 1), price1.values)
        return reg.coef_[0]

    def calculate_spread(self, price1: pd.Series, price2: pd.Series, hedge_ratio: float) -> pd.Series:
        """
Calculate

        Args:
            price1, price2: 
            hedge_ratio: 

        Returns:
            
"""
        return price1 - hedge_ratio * price2

    def fit(self, price_data: Dict[str, pd.DataFrame], **kwargs) -> None:
        """
strategyparameters

        Args:
            price_data: data {symbol: OHLCV_dataframe}
"""
        logger.info("Fitting pairs trading strategy...")

        # Handle single stock case - create synthetic pair with lagged version
        if len(price_data) < 2:
            logger.warning("Less than 2 symbols provided. Creating synthetic pair with lagged data.")
            symbol = list(price_data.keys())[0]
            data = price_data[symbol]
            close_col = 'close' if 'close' in data.columns else 'Close'

            # Create synthetic pair using lagged data
            self.pairs = [(symbol, f"{symbol}_lagged", 0.01)]  # synthetic pair
            pair_key = f"{symbol}_{symbol}_lagged"

            # Calculate synthetic hedge ratio and spread parameters
            price_series = data[close_col]
            lagged_price = price_series.shift(self.formation_period // 4)  # lag by 1/4 of formation period

            # Use simple correlation-based hedge ratio for synthetic pair
            valid_data = pd.concat([price_series, lagged_price], axis=1).dropna()
            if len(valid_data) > 20:
                corr = valid_data.iloc[:, 0].corr(valid_data.iloc[:, 1])
                self.hedge_ratios[pair_key] = max(0.5, min(1.5, corr))  # bounded hedge ratio

                # Calculate spread statistics
                spread = valid_data.iloc[:, 0] - self.hedge_ratios[pair_key] * valid_data.iloc[:, 1]
                self.spread_means[pair_key] = spread.mean()
                self.spread_stds[pair_key] = spread.std()
            else:
                # Default values if insufficient data
                self.hedge_ratios[pair_key] = 1.0
                self.spread_means[pair_key] = 0.0
                self.spread_stds[pair_key] = price_series.std() * 0.1

            self.is_fitted = True
            logger.info("Strategy fitted with synthetic pair for single symbol")
            return

        # Original multi-symbol logic
        close_prices = {}
        for symbol, data in price_data.items():
            if 'close' in data.columns:
                close_prices[symbol] = data['close']
            elif 'Close' in data.columns:
                close_prices[symbol] = data['Close']
            else:
                raise ValueError(f"No 'close' column found for {symbol}")

        #
        pairs = self.find_cointegrated_pairs(close_prices)

        if not pairs:
            logger.warning("No cointegrated pairs found in multi-symbol data")
            # Fall back to first two symbols with basic parameters
            symbols = list(close_prices.keys())[:2]
            self.pairs = [(symbols[0], symbols[1], 0.1)]
            pair_key = f"{symbols[0]}_{symbols[1]}"

            # Calculate basic parameters
            price1 = close_prices[symbols[0]][-self.formation_period:]
            price2 = close_prices[symbols[1]][-self.formation_period:]

            self.hedge_ratios[pair_key] = self._calculate_hedge_ratio(price1, price2)
            spread = self.calculate_spread(price1, price2, self.hedge_ratios[pair_key])
            self.spread_means[pair_key] = spread.mean()
            self.spread_stds[pair_key] = spread.std()
        else:
            # Save（）
            self.pairs = pairs[:5]  # 5

        # Calculateparameters
        for symbol1, symbol2, coint_score in self.pairs:
            pair_key = f"{symbol1}_{symbol2}"

            # Getdata
            price1 = close_prices[symbol1][-self.formation_period:]
            price2 = close_prices[symbol2][-self.formation_period:]

            # Calculate
            hedge_ratio = self._calculate_hedge_ratio(price1, price2)
            self.hedge_ratios[pair_key] = hedge_ratio

            # Calculate
            spread = self.calculate_spread(price1, price2, hedge_ratio)
            self.spread_means[pair_key] = spread.mean()
            self.spread_stds[pair_key] = spread.std()

            logger.info(f"Pair {pair_key}: hedge_ratio={hedge_ratio:.4f}, "
                       f"spread_mean={spread.mean():.4f}, spread_std={spread.std():.4f}")

        self.is_fitted = True
        logger.info(f"Strategy fitted with {len(self.pairs)} pairs")

    def generate_signals(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
GeneratePairs tradingsignals

        Args:
            price_data: data

        Returns:
            Containssignals、Signal strength
"""
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before generating signals")

        logger.info("Generating pairs trading signals...")

        # 
        close_prices = {}
        for symbol, data in price_data.items():
            close_prices[symbol] = data['close']

        # Generatesignals
        all_signals = {}
        all_strengths = {}
        all_confidences = {}
        pair_metadata = {}

        for symbol1, symbol2, coint_score in self.pairs:
            pair_key = f"{symbol1}_{symbol2}"

            # Handle synthetic pairs (lagged data)
            if symbol2.endswith('_lagged'):
                base_symbol = symbol1
                if base_symbol not in close_prices:
                    continue

                price1 = close_prices[base_symbol]
                price2 = close_prices[base_symbol].shift(self.formation_period // 4)  # recreate lagged data

                # CalculateWhen
                hedge_ratio = self.hedge_ratios[pair_key]
                spread = self.calculate_spread(price1, price2, hedge_ratio)
            else:
                # Regular pairs
                if symbol1 not in close_prices or symbol2 not in close_prices:
                    continue

                # Getdata
                price1 = close_prices[symbol1]
                price2 = close_prices[symbol2]

                # CalculateWhen
                hedge_ratio = self.hedge_ratios[pair_key]
                spread = self.calculate_spread(price1, price2, hedge_ratio)

            # CalculateZ-score
            spread_mean = self.spread_means[pair_key]
            spread_std = self.spread_stds[pair_key]
            zscore = (spread - spread_mean) / (spread_std + 1e-8)

            # GenerateSignal strength
            signal_strength = self._calculate_pair_signal_strength(zscore)

            # GenerateDiscrete signals
            signals = self._generate_pair_signals(zscore, pair_key)

            # Calculate
            confidence = self._calculate_pair_confidence(zscore, spread_std)

            # results
            all_signals[f"{pair_key}_signal"] = signals
            all_strengths[f"{pair_key}_strength"] = signal_strength
            all_confidences[f"{pair_key}_confidence"] = confidence

            # data
            pair_metadata[pair_key] = {
                'hedge_ratio': hedge_ratio,
                'spread': spread,
                'zscore': zscore,
                'spread_mean': spread_mean,
                'spread_std': spread_std,
                'symbol1': symbol1,
                'symbol2': symbol2
            }

        # Hassignals
        if all_signals:
            # HasSignal strength
            combined_strength = pd.concat(all_strengths.values(), axis=1).mean(axis=1)
            combined_confidence = pd.concat(all_confidences.values(), axis=1).mean(axis=1)

            # Generatesignals
            combined_signals = pd.Series(0, index=combined_strength.index)
            combined_signals[combined_strength > 0.3] = 1
            combined_signals[combined_strength < -0.3] = -1

            logger.info(f"Generated signals for {len(self.pairs)} pairs")

            return {
                'signals': combined_signals,
                'signal_strength': combined_strength,
                'confidence': combined_confidence,
                'metadata': {
                    'strategy': 'Pairs Trading',
                    'pairs': pair_metadata,
                    'individual_signals': all_signals,
                    'individual_strengths': all_strengths,
                    'individual_confidences': all_confidences
                }
            }
        else:
            # IfHasHassignals，Returnssignals
            empty_index = price_data[list(price_data.keys())[0]].index
            return {
                'signals': pd.Series(0, index=empty_index),
                'signal_strength': pd.Series(0.0, index=empty_index),
                'confidence': pd.Series(0.0, index=empty_index),
                'metadata': {'strategy': 'Pairs Trading', 'pairs': {}}
            }

    def _calculate_pair_signal_strength(self, zscore: pd.Series) -> pd.Series:
        """
CalculateSignal strength
"""
        # Signal strengthBased onZ-score，Usingtanhfunction
        strength = -np.tanh(zscore / self.entry_threshold)  # Mean reversion
        return strength

    def _generate_pair_signals(self, zscore: pd.Series, pair_key: str) -> pd.Series:
        """
Generate trading signals
"""
        signals = pd.Series(0, index=zscore.index)

        # Initialize
        current_position = 0

        for i in range(len(zscore)):
            current_zscore = zscore.iloc[i]

            if np.isnan(current_zscore):
                signals.iloc[i] = current_position
                continue

            # signals
            if abs(current_zscore) >= self.entry_threshold and current_position == 0:
                if current_zscore > 0:
                    signals.iloc[i] = -1  # ，（symbol2，symbol1）
                    current_position = -1
                else:
                    signals.iloc[i] = 1   # ，（symbol1，symbol2）
                    current_position = 1

            # signals
            elif abs(current_zscore) <= self.exit_threshold and current_position != 0:
                signals.iloc[i] = 0
                current_position = 0

            # signals
            elif abs(current_zscore) >= self.stop_loss_threshold:
                signals.iloc[i] = 0
                current_position = 0

            # When
            else:
                signals.iloc[i] = current_position

        return signals

    def _calculate_pair_confidence(self, zscore: pd.Series, spread_std: float) -> pd.Series:
        """
CalculatePairs trading
"""
        # Based onZ-score
        abs_zscore = zscore.abs()

        # Z-score，
        zscore_confidence = np.tanh(abs_zscore / self.entry_threshold)

        # ，（）
        stability_factor = min(1.0, 0.02 / (spread_std + 1e-8))  # 0.02Is

        confidence = zscore_confidence * stability_factor
        return confidence.clip(0, 1)

    def get_strategy_info(self) -> Dict[str, Any]:
        """
Getstrategy
"""
        return {
            'name': self.name,
            'type': 'Traditional - Pairs Trading',
            'parameters': {
                'formation_period': self.formation_period,
                'trading_period': self.trading_period,
                'entry_threshold': self.entry_threshold,
                'exit_threshold': self.exit_threshold,
                'stop_loss_threshold': self.stop_loss_threshold
            },
            'pairs': [(s1, s2) for s1, s2, _ in self.pairs] if self.pairs else [],
            'is_fitted': self.is_fitted
        }


if __name__ == "__main__":
    # TestPairs tradingstrategy
    from ...signal_engine.data_sources.yahoo_finance import YahooFinanceSource

    # Getdata
    source = YahooFinanceSource()
    symbols = ['AAPL', 'MSFT']  # 

    price_data = {}
    for symbol in symbols:
        try:
            data = source.download_data(symbol, period="5d", interval="1m")
            price_data[symbol] = data
            print(f"Downloaded {len(data)} records for {symbol}")
        except Exception as e:
            print(f"Failed to download data for {symbol}: {e}")

    if len(price_data) >= 2:
        # CreateRunPairs tradingstrategy
        pairs_strategy = PairsTradingStrategy(
            formation_period=100,
            entry_threshold=1.5,
            exit_threshold=0.5
        )

        try:
            # strategy
            pairs_strategy.fit(price_data)

            # Generatesignals
            results = pairs_strategy.generate_signals(price_data)

            print("\n=== Pairs Trading Strategy Results ===")
            print(f"Strategy Info: {pairs_strategy.get_strategy_info()}")
            print(f"Signal Summary:")
            print(f"  Buy signals: {sum(results['signals'] == 1)}")
            print(f"  Sell signals: {sum(results['signals'] == -1)}")
            print(f"  Hold signals: {sum(results['signals'] == 0)}")
            print(f"  Mean confidence: {results['confidence'].mean():.3f}")
            print(f"  Mean |signal strength|: {results['signal_strength'].abs().mean():.3f}")

        except Exception as e:
            print(f"Strategy execution failed: {e}")
    else:
        print("Need at least 2 symbols for pairs trading")