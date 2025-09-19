"""
Trading environment
Based onQuant_trading_systemTrading environment，HFTsignals
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class HFTTradingEnv(gym.Env):
    """
Trading environment
    Integrationsignals
"""

    def __init__(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame = None,
        signals: pd.Series = None,
        window_size: int = 10,
        initial_balance: float = 10000,
        transaction_cost: float = 0.001,
        max_position: float = 1.0
    ):
        """
InitializeTrading environment

        Args:
            data: OHLCVdata
            features: Technical indicatorsfeatures
            signals: MLsignals ()
            window_size: 
            initial_balance: 
            transaction_cost: 
            max_position: 
"""
        super().__init__()

        self.price_data = data.reset_index(drop=True)
        self.features = features.reset_index(drop=True) if features is not None else None
        self.ml_signals = signals.reset_index(drop=True) if signals is not None else None

        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position

        # Action space: continuous action [-1, 1] 
        self.action_space = gym.spaces.Box(
            low=-max_position, high=max_position, shape=(1,), dtype=np.float32
        )

        # :  + Technical indicators +  + MLsignals(IfHas)
        obs_dim = self._calculate_observation_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # variable
        self.reset()

        logger.info(f"HFT Trading Environment initialized:")
        logger.info(f"  - Data shape: {self.price_data.shape}")
        logger.info(f"  - Features: {self.features.shape if self.features is not None else None}")
        logger.info(f"  - Observation dim: {obs_dim}")
        logger.info(f"  - Action space: {self.action_space}")

    def _calculate_observation_dim(self) -> int:
        """
Calculate
"""
        #  (OHLCV normalized) * window_size
        price_dim = 5 * self.window_size

        # Technical indicatorsfeatures
        feature_dim = len(self.features.columns) if self.features is not None else 0

        # : [balance, position, unrealized_pnl, total_trades, avg_trade_return]
        account_dim = 5

        # MLsignals (IfHas)
        signal_dim = 1 if self.ml_signals is not None else 0

        return price_dim + feature_dim + account_dim + signal_dim

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """

"""
        super().reset(seed=seed)

        # 
        self.balance = self.initial_balance
        self.position = 0.0  # When [-max_position, max_position]
        self.position_value = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0

        # 
        self.total_trades = 0
        self.successful_trades = 0
        self.trade_returns = []

        # Time
        self.current_step = self.window_size
        self.max_steps = len(self.price_data) - 1

        # 
        self.trade_history = []
        self.portfolio_values = [self.initial_balance]

        # Get
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """

"""
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0, True, False, self._get_info()

        # 
        target_position = np.clip(action[0], -self.max_position, self.max_position)

        # When
        current_price = self.price_data.iloc[self.current_step]['close']

        # Calculate
        position_change = target_position - self.position

        # 
        if abs(position_change) > 0.01:  # 
            trade_cost = abs(position_change) * current_price * self.transaction_cost
            self.balance -= trade_cost

            # 
            self.trade_history.append({
                'step': self.current_step,
                'price': current_price,
                'position_change': position_change,
                'new_position': target_position,
                'cost': trade_cost
            })

            self.total_trades += 1

        # Update
        old_position = self.position
        self.position = target_position

        # Calculate
        reward = self._calculate_reward(old_position, current_price)

        # Update
        self.position_value = self.position * current_price
        total_value = self.balance + self.position_value
        self.portfolio_values.append(total_value)

        # UpdateNotImplementation
        if self.current_step > 0:
            prev_price = self.price_data.iloc[self.current_step - 1]['close']
            self.unrealized_pnl = self.position * (current_price - prev_price)

        # 
        self.current_step += 1

        # CheckIsNotEnding
        done = self.current_step >= self.max_steps
        truncated = False

        # Check
        if total_value <= 0.1 * self.initial_balance:
            done = True
            reward -= 10  # 

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, done, truncated, info

    def _calculate_reward(self, old_position: float, current_price: float) -> float:
        """
CalculateReward function
"""
        if self.current_step == 0:
            return 0

        # Get
        prev_price = self.price_data.iloc[self.current_step - 1]['close']
        price_return = (current_price - prev_price) / prev_price

        # :  * 
        position_return = old_position * price_return

        # Adjust ()
        reward = position_return * 252 * 24 * 60  # 1data

        # Adjust
        if len(self.portfolio_values) > 10:
            returns = pd.Series(self.portfolio_values).pct_change().dropna()
            if returns.std() > 0:
                sharpe_bonus = returns.mean() / returns.std() * 0.1
                reward += sharpe_bonus

        # MLsignals (IfHas)
        if self.ml_signals is not None and self.current_step < len(self.ml_signals):
            ml_signal = self.ml_signals.iloc[self.current_step]
            if (ml_signal > 0 and old_position > 0) or (ml_signal < 0 and old_position < 0):
                reward += 0.01  # signals

        return reward

    def _get_observation(self) -> np.ndarray:
        """

"""
        if self.current_step < self.window_size:
            # Processdata
            start_idx = 0
            actual_window = self.current_step + 1
        else:
            start_idx = self.current_step - self.window_size + 1
            actual_window = self.window_size

        # 1. features ()
        price_window = self.price_data.iloc[start_idx:self.current_step + 1]
        price_features = []

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in price_window.columns:
                values = price_window[col].values
                if len(values) < self.window_size:
                    # 
                    padded_values = np.concatenate([
                        np.full(self.window_size - len(values), values[0]),
                        values
                    ])
                else:
                    padded_values = values[-self.window_size:]

                #  (volume)
                if col != 'volume':
                    padded_values = padded_values / padded_values[-1] - 1  # When
                else:
                    padded_values = padded_values / (padded_values.mean() + 1e-8) - 1

                price_features.extend(padded_values)

        # 2. Technical indicatorsfeatures
        feature_values = []
        if self.features is not None and self.current_step < len(self.features):
            current_features = self.features.iloc[self.current_step]
            feature_values = current_features.fillna(0).values

        # 3. features
        total_value = self.balance + self.position_value
        account_features = [
            self.balance / self.initial_balance - 1,  # 
            self.position / self.max_position,  # 
            self.unrealized_pnl / self.initial_balance,  # NotImplementationProfit/Loss ratio
            min(self.total_trades / 100, 1),  #  ()
            np.mean(self.trade_returns) if self.trade_returns else 0  # 
        ]

        # 4. MLsignals (IfHas)
        signal_features = []
        if self.ml_signals is not None and self.current_step < len(self.ml_signals):
            signal_value = self.ml_signals.iloc[self.current_step]
            signal_features = [np.clip(signal_value, -1, 1)]  # signals

        # Hasfeatures
        observation = np.concatenate([
            price_features,
            feature_values,
            account_features,
            signal_features
        ]).astype(np.float32)

        # ProcessNaNNo
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

        return observation

    def _get_info(self) -> Dict[str, Any]:
        """
Get
"""
        total_value = self.balance + self.position_value

        return {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'position_value': self.position_value,
            'total_value': total_value,
            'total_return': (total_value - self.initial_balance) / self.initial_balance,
            'unrealized_pnl': self.unrealized_pnl,
            'total_trades': self.total_trades,
            'current_price': self.price_data.iloc[self.current_step]['close'] if self.current_step < len(self.price_data) else None
        }

    def render(self, mode: str = 'human'):
        """

"""
        info = self._get_info()
        print(f"Step: {info['step']}, Value: ${info['total_value']:.2f}, "
              f"Position: {info['position']:.3f}, Return: {info['total_return']:.2%}")

    def get_performance_metrics(self) -> Dict[str, float]:
        """
GetPerformance metrics
"""
        if len(self.portfolio_values) < 2:
            return {}

        values = pd.Series(self.portfolio_values)
        returns = values.pct_change().dropna()

        metrics = {
            'total_return': (values.iloc[-1] - values.iloc[0]) / values.iloc[0],
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252 * 24 * 60) if returns.std() > 0 else 0,
            'max_drawdown': (values / values.expanding().max() - 1).min(),
            'volatility': returns.std() * np.sqrt(252 * 24 * 60),
            'total_trades': self.total_trades,
            'win_rate': self.successful_trades / max(self.total_trades, 1)
        }

        return metrics


if __name__ == "__main__":
    # Test
    from ...signal_engine.signal_processor import SignalProcessor

    # UsingSignal processorGetdatasignals
    processor = SignalProcessor()
    results = processor.run_full_pipeline(symbol="AAPL", period="2d", interval="1m")

    # Create
    env = HFTTradingEnv(
        data=results['raw_data'],
        features=results['features'],
        signals=results['signals']
    )

    # Test
    obs, info = env.reset()
    print(f"Environment initialized. Observation shape: {obs.shape}")

    # Test
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()

        if done:
            break

    # performance
    print("\n=== Performance Metrics ===")
    for k, v in env.get_performance_metrics().items():
        print(f"{k}: {v:.4f}")