"""
Reinforcement LearningAgent
PPO, DQNSACRLForHFT
"""

import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path

# Stable-Baselines3 imports
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from .trading_env import HFTTradingEnv

logger = logging.getLogger(__name__)


class TradingCallback(BaseCallback):
    """
Trainfunction
"""

    def __init__(self, eval_env, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.eval_rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluatemodel
            obs = self.eval_env.reset()[0]
            episode_reward = 0
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                if truncated:
                    break

            self.eval_rewards.append(episode_reward)

            if self.verbose > 0:
                print(f"Eval at step {self.n_calls}: Episode reward = {episode_reward:.2f}")

            # Savemodel
            mean_reward = np.mean(self.eval_rewards[-10:])  # 10
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"New best mean reward: {mean_reward:.2f}")

        return True


class RLAgentManager:
    """
Reinforcement LearningAgent
"""

    def __init__(self, env: HFTTradingEnv):
        """
InitializeRLAgent

        Args:
            env: Trading environment
"""
        self.env = env
        self.agents = {}
        self.training_history = {}

        # Supportsconfig
        self.algorithm_configs = {
            'ppo': {
                'class': PPO,
                'policy': 'MlpPolicy',
                'default_params': {
                    'learning_rate': 3e-4,
                    'n_steps': 2048,
                    'batch_size': 64,
                    'n_epochs': 10,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_range': 0.2,
                    'ent_coef': 0.01,
                    'vf_coef': 0.5,
                    'max_grad_norm': 0.5,
                    'verbose': 1
                }
            },
            'dqn': {
                'class': DQN,
                'policy': 'MlpPolicy',
                'default_params': {
                    'learning_rate': 1e-4,
                    'batch_size': 32,
                    'buffer_size': 50000,
                    'learning_starts': 1000,
                    'target_update_interval': 500,
                    'train_freq': 4,
                    'gradient_steps': 1,
                    'exploration_fraction': 0.1,
                    'exploration_initial_eps': 1.0,
                    'exploration_final_eps': 0.05,
                    'verbose': 1
                }
            },
            'sac': {
                'class': SAC,
                'policy': 'MlpPolicy',
                'default_params': {
                    'learning_rate': 3e-4,
                    'buffer_size': 100000,
                    'batch_size': 256,
                    'tau': 0.005,
                    'gamma': 0.99,
                    'train_freq': 1,
                    'gradient_steps': 1,
                    'learning_starts': 1000,
                    'verbose': 1
                }
            }
        }

        logger.info("RL Agent Manager initialized")

    def create_agent(
        self,
        algorithm: str,
        **kwargs
    ) -> Any:
        """
CreateRLAgent

        Args:
            algorithm:  ('ppo', 'dqn', 'sac')
            **kwargs: Additional parameters

        Returns:
            TrainAgent
"""
        if algorithm not in self.algorithm_configs:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        config = self.algorithm_configs[algorithm]
        agent_class = config['class']
        policy = config['policy']

        # parametersparameters
        params = config['default_params'].copy()
        params.update(kwargs)

        # DQN，RequiresAction space
        env = self.env
        if algorithm == 'dqn':
            # DQNRequiresAction space，RequiresPackage
            logger.warning("DQN requires discrete action space. Consider using PPO or SAC for continuous actions.")

        # Create
        vec_env = DummyVecEnv([lambda: Monitor(env)])

        # CreateAgent
        agent = agent_class(policy, vec_env, **params)

        self.agents[algorithm] = agent
        logger.info(f"Created {algorithm.upper()} agent with parameters: {params}")

        return agent

    def train_agent(
        self,
        algorithm: str,
        total_timesteps: int = 50000,
        callback: Optional[BaseCallback] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
TrainAgent

        Args:
            algorithm: 
            total_timesteps: Train
            callback: function
            **kwargs: Additional parameters

        Returns:
            Training results
"""
        if algorithm not in self.agents:
            logger.info(f"Agent {algorithm} not found. Creating new agent...")
            self.create_agent(algorithm, **kwargs)

        agent = self.agents[algorithm]

        logger.info(f"Training {algorithm.upper()} agent for {total_timesteps} timesteps...")

        # CreateEvaluateFor
        if callback is None:
            eval_env = Monitor(self.env)
            callback = TradingCallback(eval_env, eval_freq=max(1000, total_timesteps // 10))

        # Train
        try:
            agent.learn(total_timesteps=total_timesteps, callback=callback)

            # SaveTrain
            self.training_history[algorithm] = {
                'total_timesteps': total_timesteps,
                'eval_rewards': callback.eval_rewards if hasattr(callback, 'eval_rewards') else [],
                'best_reward': callback.best_mean_reward if hasattr(callback, 'best_mean_reward') else 0
            }

            logger.info(f"Training completed for {algorithm.upper()}")

            return self.training_history[algorithm]

        except Exception as e:
            logger.error(f"Training failed for {algorithm}: {str(e)}")
            raise

    def evaluate_agent(
        self,
        algorithm: str,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """
EvaluateAgentperformance

        Args:
            algorithm: 
            n_episodes: Evaluate
            deterministic: IsNotUsingstrategy

        Returns:
            Evaluateresults
"""
        if algorithm not in self.agents:
            raise ValueError(f"Agent {algorithm} not found")

        agent = self.agents[algorithm]
        episode_rewards = []
        episode_metrics = []

        logger.info(f"Evaluating {algorithm.upper()} agent for {n_episodes} episodes...")

        for episode in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _ = agent.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward

                if truncated:
                    break

            episode_rewards.append(episode_reward)
            episode_metrics.append(self.env.get_performance_metrics())

        # Calculatemetrics
        results = {
            'algorithm': algorithm,
            'n_episodes': n_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'episode_rewards': episode_rewards
        }

        # CalculatePerformance metrics
        if episode_metrics:
            avg_metrics = {}
            for key in episode_metrics[0].keys():
                avg_metrics[f'avg_{key}'] = np.mean([m[key] for m in episode_metrics])
            results.update(avg_metrics)

        logger.info(f"Evaluation completed. Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")

        return results

    def compare_agents(
        self,
        algorithms: List[str] = None,
        n_episodes: int = 10
    ) -> pd.DataFrame:
        """
CompareAgentperformance

        Args:
            algorithms: Compare
            n_episodes: Evaluate

        Returns:
            CompareresultsDataFrame
"""
        if algorithms is None:
            algorithms = list(self.agents.keys())

        comparison_results = []

        for algorithm in algorithms:
            if algorithm in self.agents:
                try:
                    results = self.evaluate_agent(algorithm, n_episodes)
                    comparison_results.append(results)
                except Exception as e:
                    logger.error(f"Failed to evaluate {algorithm}: {str(e)}")

        if not comparison_results:
            return pd.DataFrame()

        # DataFrame
        df = pd.DataFrame(comparison_results)

        # 
        display_columns = [
            'algorithm', 'mean_reward', 'std_reward',
            'avg_total_return', 'avg_sharpe_ratio', 'avg_max_drawdown'
        ]
        available_columns = [col for col in display_columns if col in df.columns]

        return df[available_columns].sort_values('mean_reward', ascending=False)

    def save_agent(self, algorithm: str, filepath: str) -> None:
        """
SaveAgentmodel
"""
        if algorithm not in self.agents:
            raise ValueError(f"Agent {algorithm} not found")

        self.agents[algorithm].save(filepath)
        logger.info(f"Agent {algorithm} saved to {filepath}")

    def load_agent(self, algorithm: str, filepath: str) -> None:
        """
LoadAgentmodel
"""
        if algorithm not in self.algorithm_configs:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        config = self.algorithm_configs[algorithm]
        agent_class = config['class']

        # Create (LoadRequires)
        vec_env = DummyVecEnv([lambda: Monitor(self.env)])

        # LoadAgent
        agent = agent_class.load(filepath, env=vec_env)
        self.agents[algorithm] = agent

        logger.info(f"Agent {algorithm} loaded from {filepath}")

    def get_trading_signals(
        self,
        algorithm: str,
        features: pd.DataFrame,
        deterministic: bool = True
    ) -> pd.Series:
        """
UsingTrainAgentGenerate trading signals

        Args:
            algorithm: 
            features: Inputfeatures
            deterministic: IsNotUsingstrategy

        Returns:
            signals
"""
        if algorithm not in self.agents:
            raise ValueError(f"Agent {algorithm} not found")

        agent = self.agents[algorithm]
        signals = []

        # data
        # Note: RequiresAccording toAdjustdata
        for idx in range(len(features)):
            try:
                # Get (Process，)
                obs = self.env._get_observation()
                action, _ = agent.predict(obs, deterministic=deterministic)

                # signals (-1, 0, 1)
                if isinstance(action, np.ndarray):
                    signal = 1 if action[0] > 0.1 else (-1 if action[0] < -0.1 else 0)
                else:
                    signal = action

                signals.append(signal)

            except Exception as e:
                logger.warning(f"Error generating signal at step {idx}: {str(e)}")
                signals.append(0)

        return pd.Series(signals, index=features.index)


if __name__ == "__main__":
    # TestRLAgent
    from ...signal_engine.signal_processor import SignalProcessor

    # Getdata
    processor = SignalProcessor()
    results = processor.run_full_pipeline(symbol="AAPL", period="2d", interval="1m")

    # Create
    env = HFTTradingEnv(
        data=results['raw_data'],
        features=results['features'],
        signals=results['signals']
    )

    # CreateRL
    rl_manager = RLAgentManager(env)

    # TrainPPOAgent
    print("Training PPO agent...")
    rl_manager.train_agent('ppo', total_timesteps=10000)

    # EvaluateAgent
    print("\nEvaluating PPO agent...")
    eval_results = rl_manager.evaluate_agent('ppo', n_episodes=5)
    print(f"Mean reward: {eval_results['mean_reward']:.2f}")

    # Generate trading signals
    print("\nGenerating trading signals...")
    signals = rl_manager.get_trading_signals('ppo', results['features'])
    print(f"Generated {len(signals)} signals")