"""
Market Regime Classification Module
==================================

Classifies market into 9 regimes based on three dimensions:
- Liquidity (spread-based, depth-based indicators)
- Volume (trading intensity)
- Volatility (price movement patterns)

9 Regime Matrix:
High Liquidity:    High Volume + High Vol | High Volume + Low Vol | Low Volume + High Vol  | Low Volume + Low Vol
Medium Liquidity:  High Volume + High Vol | High Volume + Low Vol | Low Volume + High Vol  | Low Volume + Low Vol
Low Liquidity:     High Volume + High Vol | High Volume + Low Vol | Low Volume + High Vol  | Low Volume + Low Vol

Each regime has different characteristics for Frequentist vs Bayesian approach effectiveness.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class LiquidityLevel(Enum):
    """Liquidity classification levels"""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class VolumeLevel(Enum):
    """Volume classification levels"""
    HIGH = "High"
    LOW = "Low"


class VolatilityLevel(Enum):
    """Volatility classification levels"""
    HIGH = "High"
    LOW = "Low"


@dataclass
class MarketRegime:
    """Market regime data class"""
    liquidity: LiquidityLevel
    volume: VolumeLevel
    volatility: VolatilityLevel
    name: str
    characteristics: str
    frequentist_advantage: str
    bayesian_advantage: str
    recommended_approach: str


class MarketRegimeClassifier:
    """
    Market Regime Classifier for HFT Trading

    Classifies market conditions into 9 distinct regimes based on:
    1. Liquidity metrics (spread, depth, order book thickness)
    2. Volume intensity (trading frequency, volume patterns)
    3. Volatility patterns (price movement characteristics)
    """

    def __init__(self,
                 lookback_period: int = 100,
                 liquidity_thresholds: Tuple[float, float] = (0.33, 0.67),
                 volume_threshold: float = 0.5,
                 volatility_threshold: float = 0.5):
        """
        Initialize Market Regime Classifier

        Args:
            lookback_period: Number of periods for rolling calculations
            liquidity_thresholds: (low, high) thresholds for liquidity terciles
            volume_threshold: Threshold for volume binary classification
            volatility_threshold: Threshold for volatility binary classification
        """
        self.lookback_period = lookback_period
        self.liquidity_thresholds = liquidity_thresholds
        self.volume_threshold = volume_threshold
        self.volatility_threshold = volatility_threshold

        # Define the 9 market regimes
        self.regimes = self._initialize_regimes()

        logger.info(f"Market Regime Classifier initialized with {len(self.regimes)} regimes")

    def _initialize_regimes(self) -> Dict[str, MarketRegime]:
        """Initialize the 9 market regimes with their characteristics"""

        regimes = {
            "HighLiq_HighVol_HighVol": MarketRegime(
                liquidity=LiquidityLevel.HIGH,
                volume=VolumeLevel.HIGH,
                volatility=VolatilityLevel.HIGH,
                name="Deep Active Volatile",
                characteristics="深度足，交易活跃，但波动大",
                frequentist_advantage="大样本下参数估计稳健；可用GARCH/Hawkes拟合",
                bayesian_advantage="可建跳跃/状态切换先验，量化尾部风险",
                recommended_approach="Bayesian → 不确定性更关键"
            ),

            "HighLiq_HighVol_LowVol": MarketRegime(
                liquidity=LiquidityLevel.HIGH,
                volume=VolumeLevel.HIGH,
                volatility=VolatilityLevel.LOW,
                name="Deep Active Stable",
                characteristics="深度足，稳定大盘 (BTC/ETH)",
                frequentist_advantage="参数估计快、稳；显著性检验可靠",
                bayesian_advantage="不确定性作用小",
                recommended_approach="Frequentist 更高效"
            ),

            "HighLiq_LowVol_HighVol": MarketRegime(
                liquidity=LiquidityLevel.HIGH,
                volume=VolumeLevel.LOW,
                volatility=VolatilityLevel.HIGH,
                name="Deep Quiet Volatile",
                characteristics="深度足但成交稀疏，噪声大",
                frequentist_advantage="样本不足，估计偏差大",
                bayesian_advantage="层次贝叶斯可引入先验，缓解稀疏数据问题",
                recommended_approach="Bayesian 更优"
            ),

            "HighLiq_LowVol_LowVol": MarketRegime(
                liquidity=LiquidityLevel.HIGH,
                volume=VolumeLevel.LOW,
                volatility=VolatilityLevel.LOW,
                name="Deep Quiet Stable",
                characteristics="冷清但盘口稳",
                frequentist_advantage="检验结果偏保守，但还能运行",
                bayesian_advantage="先验可能过拟合冷清市场",
                recommended_approach="Frequentist 简单够用"
            ),

            "MedLiq_HighVol_HighVol": MarketRegime(
                liquidity=LiquidityLevel.MEDIUM,
                volume=VolumeLevel.HIGH,
                volatility=VolatilityLevel.HIGH,
                name="Medium Active Volatile",
                characteristics="主流币常见状态，适度活跃",
                frequentist_advantage="大样本估计稳健",
                bayesian_advantage="不确定性量化有价值",
                recommended_approach="混合：Frequentist 打底 + Bayesian 调优"
            ),

            "MedLiq_HighVol_LowVol": MarketRegime(
                liquidity=LiquidityLevel.MEDIUM,
                volume=VolumeLevel.HIGH,
                volatility=VolatilityLevel.LOW,
                name="Medium Active Stable",
                characteristics="中等流动性，活跃但稳定",
                frequentist_advantage="足够样本，参数估计可靠",
                bayesian_advantage="可提供额外的不确定性信息",
                recommended_approach="两者都可，Frequentist 优先"
            ),

            "MedLiq_LowVol_HighVol": MarketRegime(
                liquidity=LiquidityLevel.MEDIUM,
                volume=VolumeLevel.LOW,
                volatility=VolatilityLevel.HIGH,
                name="Medium Quiet Volatile",
                characteristics="中等流动性，交易稀少但波动大",
                frequentist_advantage="样本不足导致估计不稳定",
                bayesian_advantage="先验知识可以稳定估计",
                recommended_approach="Bayesian 更可靠"
            ),

            "LowLiq_HighVol_HighVol": MarketRegime(
                liquidity=LiquidityLevel.LOW,
                volume=VolumeLevel.HIGH,
                volatility=VolatilityLevel.HIGH,
                name="Shallow Active Volatile",
                characteristics="小币种爆发，成交多但盘口薄",
                frequentist_advantage="高频统计显著性常失效",
                bayesian_advantage="动态先验能适应regime shift",
                recommended_approach="Bayesian 更稳健"
            ),

            "LowLiq_LowVol_HighVol": MarketRegime(
                liquidity=LiquidityLevel.LOW,
                volume=VolumeLevel.LOW,
                volatility=VolatilityLevel.HIGH,
                name="Shallow Quiet Volatile",
                characteristics="山寨币典型，数据稀少且剧烈跳动",
                frequentist_advantage="检验基本失效，易假信号",
                bayesian_advantage="Bayesian通过先验建模稀疏数据",
                recommended_approach="Bayesian 明显更好"
            )
        }

        return regimes

    def calculate_liquidity_metrics(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate liquidity metrics

        For real market data, this would use bid-ask spread, order book depth, etc.
        For simplified analysis with OHLCV data, we use price range and volume patterns.

        Args:
            data: OHLCV DataFrame

        Returns:
            Series with liquidity scores (0-1, higher = more liquid)
        """
        # Calculate various liquidity proxies

        # 1. Relative spread proxy (using high-low range as spread proxy)
        spread_proxy = (data['high'] - data['low']) / data['close']

        # 2. Volume consistency (more consistent volume = more liquid)
        volume_cv = data['volume'].rolling(self.lookback_period).std() / \
                   data['volume'].rolling(self.lookback_period).mean()

        # 3. Price impact proxy (smaller price moves per unit volume = more liquid)
        returns = data['close'].pct_change().abs()
        volume_normalized = data['volume'] / data['volume'].rolling(self.lookback_period).mean()
        price_impact = returns / (volume_normalized + 1e-10)  # Avoid division by zero

        # Combine metrics (invert so higher = more liquid)
        liquidity_score = (
            1 / (1 + spread_proxy.rolling(self.lookback_period).mean()) * 0.4 +
            1 / (1 + volume_cv.fillna(1)) * 0.3 +
            1 / (1 + price_impact.rolling(self.lookback_period).mean()) * 0.3
        )

        return liquidity_score.fillna(0.5)  # Default to medium liquidity

    def calculate_volume_metrics(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate volume intensity metrics

        Args:
            data: OHLCV DataFrame

        Returns:
            Series with volume scores (0-1, higher = higher volume)
        """
        # Volume relative to recent average
        volume_sma = data['volume'].rolling(self.lookback_period).mean()
        relative_volume = data['volume'] / volume_sma

        # Volume trend (increasing volume = more active)
        volume_trend = data['volume'].rolling(20).apply(
            lambda x: np.corrcoef(np.arange(len(x)), x)[0,1] if len(x) == 20 else 0
        )

        # Combine metrics
        volume_score = (
            np.tanh(relative_volume - 1) * 0.5 + 0.5  # Normalize to 0-1
        ) * 0.7 + (volume_trend + 1) * 0.5 * 0.3  # Add trend component

        return volume_score.fillna(0.5)

    def calculate_volatility_metrics(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate volatility metrics

        Args:
            data: OHLCV DataFrame

        Returns:
            Series with volatility scores (0-1, higher = more volatile)
        """
        # Calculate returns
        returns = data['close'].pct_change()

        # Rolling standard deviation
        rolling_vol = returns.rolling(self.lookback_period).std()

        # Realized volatility (using high-low-close)
        # Garman-Klass volatility estimator
        gk_vol = np.sqrt(
            0.5 * np.log(data['high'] / data['low'])**2 -
            (2*np.log(2) - 1) * np.log(data['close'] / data['open'])**2
        ).rolling(self.lookback_period).mean()

        # Jump detection (large price moves)
        jump_threshold = rolling_vol * 2  # 2-sigma threshold
        jumps = (returns.abs() > jump_threshold).rolling(20).sum() / 20

        # Combine metrics
        vol_score = (
            rolling_vol / rolling_vol.rolling(self.lookback_period*2).quantile(0.8) * 0.5 +
            gk_vol / gk_vol.rolling(self.lookback_period*2).quantile(0.8) * 0.3 +
            jumps * 0.2
        )

        return vol_score.fillna(0.5).clip(0, 2)  # Cap at 2 for normalization

    def classify_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Classify market regime for each time period

        Args:
            data: OHLCV DataFrame with datetime index

        Returns:
            DataFrame with regime classifications and metrics
        """
        logger.info("Calculating market regime metrics...")

        # Calculate base metrics
        liquidity_scores = self.calculate_liquidity_metrics(data)
        volume_scores = self.calculate_volume_metrics(data)
        volatility_scores = self.calculate_volatility_metrics(data)

        # Classify into discrete levels
        regime_data = pd.DataFrame(index=data.index)
        regime_data['liquidity_score'] = liquidity_scores
        regime_data['volume_score'] = volume_scores
        regime_data['volatility_score'] = volatility_scores

        # Liquidity: 3 levels (High/Medium/Low)
        liquidity_quantiles = liquidity_scores.quantile([self.liquidity_thresholds[0], self.liquidity_thresholds[1]])
        regime_data['liquidity_level'] = np.select(
            [
                liquidity_scores <= liquidity_quantiles.iloc[0],
                liquidity_scores <= liquidity_quantiles.iloc[1]
            ],
            [LiquidityLevel.LOW.value, LiquidityLevel.MEDIUM.value],
            default=LiquidityLevel.HIGH.value
        )

        # Volume: 2 levels (High/Low)
        volume_median = volume_scores.quantile(self.volume_threshold)
        regime_data['volume_level'] = np.where(
            volume_scores >= volume_median,
            VolumeLevel.HIGH.value,
            VolumeLevel.LOW.value
        )

        # Volatility: 2 levels (High/Low)
        vol_median = volatility_scores.quantile(self.volatility_threshold)
        regime_data['volatility_level'] = np.where(
            volatility_scores >= vol_median,
            VolatilityLevel.HIGH.value,
            VolatilityLevel.LOW.value
        )

        # Create regime names
        regime_data['regime_name'] = (
            regime_data['liquidity_level'].str.replace('High', 'HighLiq').str.replace('Medium', 'MedLiq').str.replace('Low', 'LowLiq') + '_' +
            regime_data['volume_level'] + 'Vol_' +
            regime_data['volatility_level'] + 'Vol'
        )

        # Add regime characteristics
        regime_data['characteristics'] = regime_data['regime_name'].map(
            {name: regime.characteristics for name, regime in self.regimes.items()}
        )

        regime_data['recommended_approach'] = regime_data['regime_name'].map(
            {name: regime.recommended_approach for name, regime in self.regimes.items()}
        )

        logger.info(f"Classified {len(regime_data)} periods into market regimes")
        logger.info(f"Regime distribution:\n{regime_data['regime_name'].value_counts()}")

        return regime_data

    def get_regime_summary(self) -> pd.DataFrame:
        """
        Get summary of all 9 regimes with their characteristics

        Returns:
            DataFrame with regime summary
        """
        regime_summary = []

        for name, regime in self.regimes.items():
            regime_summary.append({
                'Regime': name,
                'Name': regime.name,
                'Liquidity': regime.liquidity.value,
                'Volume': regime.volume.value,
                'Volatility': regime.volatility.value,
                'Characteristics': regime.characteristics,
                'Frequentist_Advantage': regime.frequentist_advantage,
                'Bayesian_Advantage': regime.bayesian_advantage,
                'Recommended': regime.recommended_approach
            })

        return pd.DataFrame(regime_summary)

    def analyze_regime_transitions(self, regime_data: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze regime transition patterns

        Args:
            regime_data: Output from classify_regime()

        Returns:
            Dictionary with transition analysis
        """
        transitions = {}
        regime_series = regime_data['regime_name']

        # Calculate transition matrix
        transition_counts = pd.crosstab(
            regime_series.shift(1),
            regime_series,
            dropna=True
        )

        transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)

        # Calculate regime persistence (probability of staying in same regime)
        persistence = np.diag(transition_probs) if len(transition_probs) > 0 else {}

        # Average regime duration
        regime_durations = {}
        for regime in regime_series.unique():
            regime_periods = (regime_series == regime).astype(int)
            # Find consecutive periods
            regime_groups = (regime_periods != regime_periods.shift()).cumsum()
            durations = regime_periods.groupby(regime_groups).sum()
            regime_durations[regime] = durations[durations > 0].mean()

        transitions['transition_matrix'] = transition_probs
        transitions['persistence'] = persistence
        transitions['average_duration'] = regime_durations

        return transitions