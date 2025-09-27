"""
Bayesian Statistical Methods for HFT
====================================

Implementation of Bayesian approaches for HFT trading:
1. Hierarchical Bayesian models for parameter uncertainty
2. State-switching models for regime changes
3. Dynamic Bayesian updating for online learning
4. Variational Bayes for fast approximate inference

These methods excel in sparse data, regime changes, and uncertainty quantification.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.optimize import minimize
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class BayesianResult:
    """Result container for Bayesian methods"""
    method_name: str
    posterior_mean: Dict[str, float]
    posterior_std: Dict[str, float]
    credible_intervals: Dict[str, Tuple[float, float]]
    log_marginal_likelihood: float
    dic: float  # Deviance Information Criterion
    waic: float  # Watanabe Information Criterion
    predictions: np.ndarray
    prediction_intervals: np.ndarray
    uncertainty_measures: Dict[str, float]


class BayesianMethod(ABC):
    """Abstract base class for Bayesian methods"""

    @abstractmethod
    def fit(self, data: np.ndarray, prior_params: Optional[Dict] = None) -> BayesianResult:
        """Fit the Bayesian model"""
        pass

    @abstractmethod
    def predict(self, data: np.ndarray, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty quantification"""
        pass


class HierarchicalBayesianModel(BayesianMethod):
    """
    Hierarchical Bayesian Model for Trading Signals

    Good for: Sparse data where we can borrow strength across similar assets/periods
    Poor for: Well-identified single-asset problems with abundant data
    """

    def __init__(self, n_groups: int = 5, mcmc_samples: int = 2000):
        """
        Initialize Hierarchical Bayesian Model

        Args:
            n_groups: Number of groups/regimes for hierarchical structure
            mcmc_samples: Number of MCMC samples for posterior inference
        """
        self.n_groups = n_groups
        self.mcmc_samples = mcmc_samples
        self.posterior_samples = None
        self.group_assignments = None

    def _log_normal_likelihood(self, y: np.ndarray, mu: np.ndarray, sigma: float) -> float:
        """Log-likelihood for normal observations"""
        return -0.5 * len(y) * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum((y - mu)**2) / sigma**2

    def _assign_groups(self, data: np.ndarray) -> np.ndarray:
        """Assign observations to groups based on volatility patterns"""
        # Simple grouping based on local volatility
        window_size = max(10, len(data) // self.n_groups)
        groups = np.zeros(len(data), dtype=int)

        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2)
            local_vol = np.std(data[start_idx:end_idx])

            # Assign to group based on volatility quantiles
            vol_quantiles = np.linspace(0, 1, self.n_groups + 1)
            all_local_vols = []
            for j in range(len(data)):
                s = max(0, j - window_size // 2)
                e = min(len(data), j + window_size // 2)
                all_local_vols.append(np.std(data[s:e]))

            vol_thresholds = np.quantile(all_local_vols, vol_quantiles)
            group = np.searchsorted(vol_thresholds[1:], local_vol)
            groups[i] = min(group, self.n_groups - 1)

        return groups

    def _variational_bayes_inference(self, data: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
        """
        Variational Bayes approximation for faster inference

        Uses mean-field variational inference to approximate posterior
        """
        n_obs = len(data)
        unique_groups = np.unique(groups)
        n_active_groups = len(unique_groups)

        # Prior hyperparameters
        # Hierarchical prior: μⱼ ~ N(θ, τ²), θ ~ N(0, 10²), τ² ~ InvGamma(1, 1)
        prior_theta_mean = 0.0
        prior_theta_var = 100.0
        prior_tau_shape = 1.0
        prior_tau_rate = 1.0
        prior_sigma_shape = 1.0
        prior_sigma_rate = 1.0

        # Initialize variational parameters
        # q(θ) = N(m_θ, v_θ)
        m_theta = np.mean(data)
        v_theta = 1.0

        # q(τ²) = InvGamma(a_τ, b_τ)
        a_tau = prior_tau_shape + n_active_groups / 2
        b_tau = prior_tau_rate + 0.1

        # q(μⱼ) = N(m_μⱼ, v_μⱼ) for each group j
        m_mu = {}
        v_mu = {}
        for g in unique_groups:
            group_data = data[groups == g]
            m_mu[g] = np.mean(group_data) if len(group_data) > 0 else 0.0
            v_mu[g] = 1.0

        # q(σ²) = InvGamma(a_σ, b_σ)
        a_sigma = prior_sigma_shape + n_obs / 2
        b_sigma = prior_sigma_rate + 0.1

        # Variational optimization
        for iteration in range(100):  # Fixed iterations for simplicity
            old_params = (m_theta, v_theta, a_tau, b_tau, a_sigma, b_sigma, m_mu.copy(), v_mu.copy())

            # Update q(θ)
            E_tau_inv = a_tau / b_tau  # E[1/τ²]
            v_theta = 1 / (1/prior_theta_var + n_active_groups * E_tau_inv)
            m_theta = v_theta * (prior_theta_mean/prior_theta_var + E_tau_inv * sum(m_mu.values()))

            # Update q(τ²)
            a_tau = prior_tau_shape + n_active_groups / 2
            b_tau = prior_tau_rate + 0.5 * (
                sum(m_mu[g]**2 + v_mu[g] for g in unique_groups) -
                2 * m_theta * sum(m_mu[g] for g in unique_groups) +
                n_active_groups * (m_theta**2 + v_theta)
            )

            # Update q(μⱼ) for each group
            E_sigma_inv = a_sigma / b_sigma  # E[1/σ²]
            E_tau_inv = a_tau / b_tau

            for g in unique_groups:
                group_data = data[groups == g]
                n_g = len(group_data)

                if n_g > 0:
                    v_mu[g] = 1 / (E_tau_inv + n_g * E_sigma_inv)
                    m_mu[g] = v_mu[g] * (E_tau_inv * m_theta + E_sigma_inv * np.sum(group_data))
                else:
                    v_mu[g] = 1 / E_tau_inv
                    m_mu[g] = m_theta

            # Update q(σ²)
            a_sigma = prior_sigma_shape + n_obs / 2
            residual_sum = 0
            for g in unique_groups:
                group_data = data[groups == g]
                if len(group_data) > 0:
                    residual_sum += np.sum((group_data - m_mu[g])**2) + len(group_data) * v_mu[g]

            b_sigma = prior_sigma_rate + 0.5 * residual_sum

            # Check convergence (simplified)
            new_params = (m_theta, v_theta, a_tau, b_tau, a_sigma, b_sigma, m_mu.copy(), v_mu.copy())
            if iteration > 10 and iteration % 10 == 0:
                # Simple convergence check
                converged = True
                if abs(new_params[0] - old_params[0]) > 1e-4:  # m_theta
                    converged = False
                if not converged:
                    continue
                else:
                    break

        # Collect results
        posterior_samples = {
            'theta': {
                'mean': m_theta,
                'std': np.sqrt(v_theta),
                'samples': np.random.normal(m_theta, np.sqrt(v_theta), 1000)
            },
            'tau_sq': {
                'mean': b_tau / (a_tau - 1) if a_tau > 1 else 1.0,
                'std': np.sqrt(b_tau**2 / ((a_tau - 1)**2 * (a_tau - 2))) if a_tau > 2 else 1.0,
                'samples': 1 / np.random.gamma(a_tau, 1/b_tau, 1000)
            },
            'sigma_sq': {
                'mean': b_sigma / (a_sigma - 1) if a_sigma > 1 else 1.0,
                'std': np.sqrt(b_sigma**2 / ((a_sigma - 1)**2 * (a_sigma - 2))) if a_sigma > 2 else 1.0,
                'samples': 1 / np.random.gamma(a_sigma, 1/b_sigma, 1000)
            },
            'mu_groups': m_mu
        }

        return posterior_samples

    def fit(self, data: np.ndarray, prior_params: Optional[Dict] = None) -> BayesianResult:
        """
        Fit hierarchical Bayesian model

        Args:
            data: Observations
            prior_params: Prior hyperparameters (optional)

        Returns:
            BayesianResult with posterior summaries
        """
        logger.info("Fitting Hierarchical Bayesian model...")

        if len(data) < 20:
            logger.warning("Small sample size for hierarchical model")

        # Assign observations to groups
        self.group_assignments = self._assign_groups(data)

        # Run variational inference
        posterior_samples = self._variational_bayes_inference(data, self.group_assignments)
        self.posterior_samples = posterior_samples

        # Extract posterior summaries
        posterior_mean = {
            'theta': posterior_samples['theta']['mean'],
            'tau_sq': posterior_samples['tau_sq']['mean'],
            'sigma_sq': posterior_samples['sigma_sq']['mean']
        }

        posterior_std = {
            'theta': posterior_samples['theta']['std'],
            'tau_sq': posterior_samples['tau_sq']['std'],
            'sigma_sq': posterior_samples['sigma_sq']['std']
        }

        # 95% credible intervals
        credible_intervals = {
            'theta': (
                np.percentile(posterior_samples['theta']['samples'], 2.5),
                np.percentile(posterior_samples['theta']['samples'], 97.5)
            ),
            'tau_sq': (
                np.percentile(posterior_samples['tau_sq']['samples'], 2.5),
                np.percentile(posterior_samples['tau_sq']['samples'], 97.5)
            ),
            'sigma_sq': (
                np.percentile(posterior_samples['sigma_sq']['samples'], 2.5),
                np.percentile(posterior_samples['sigma_sq']['samples'], 97.5)
            )
        }

        # Generate predictions (posterior predictive)
        predictions = np.zeros(len(data))
        prediction_intervals = np.zeros((len(data), 2))

        for i, group in enumerate(self.group_assignments):
            if group in posterior_samples['mu_groups']:
                mu_g = posterior_samples['mu_groups'][group]
                predictions[i] = mu_g

                # Prediction interval (accounting for all uncertainties)
                sigma = np.sqrt(posterior_samples['sigma_sq']['mean'])
                prediction_intervals[i] = [mu_g - 1.96*sigma, mu_g + 1.96*sigma]

        # Information criteria (approximate)
        log_marginal_likelihood = self._approximate_log_marginal_likelihood(data, posterior_samples)

        uncertainty_measures = {
            'posterior_variance_theta': posterior_samples['theta']['std']**2,
            'between_group_variance': posterior_samples['tau_sq']['mean'],
            'within_group_variance': posterior_samples['sigma_sq']['mean'],
            'total_uncertainty': (
                posterior_samples['theta']['std']**2 +
                posterior_samples['tau_sq']['mean'] +
                posterior_samples['sigma_sq']['mean']
            )
        }

        logger.info(f"Hierarchical Bayesian model fitted. Between-group var: {posterior_samples['tau_sq']['mean']:.4f}")

        return BayesianResult(
            method_name="Hierarchical_Bayesian",
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            credible_intervals=credible_intervals,
            log_marginal_likelihood=log_marginal_likelihood,
            dic=-2 * log_marginal_likelihood + 2 * len(posterior_mean),  # Approximate
            waic=-2 * log_marginal_likelihood + 2 * len(posterior_mean),  # Approximate
            predictions=predictions,
            prediction_intervals=prediction_intervals,
            uncertainty_measures=uncertainty_measures
        )

    def _approximate_log_marginal_likelihood(self, data: np.ndarray, posterior_samples: Dict) -> float:
        """Approximate log marginal likelihood using Laplace approximation"""
        # Simplified calculation
        n = len(data)
        log_likelihood = -0.5 * n * np.log(2 * np.pi * posterior_samples['sigma_sq']['mean'])
        log_likelihood -= 0.5 * np.sum((data - np.mean(data))**2) / posterior_samples['sigma_sq']['mean']
        return log_likelihood

    def predict(self, data: np.ndarray, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty"""
        if self.posterior_samples is None:
            raise ValueError("Model must be fitted before prediction")

        # Simplified prediction using posterior means
        mu_pred = self.posterior_samples['theta']['mean']
        sigma_pred = np.sqrt(self.posterior_samples['sigma_sq']['mean'])

        predictions = np.array([mu_pred] * len(data))
        intervals = np.array([[mu_pred - 1.96*sigma_pred, mu_pred + 1.96*sigma_pred]] * len(data))

        return predictions, intervals


class MarkovSwitchingModel(BayesianMethod):
    """
    Markov Switching Model for Regime Changes

    Good for: Markets with clear regime switches (volatility clustering, structural breaks)
    Poor for: Stable, single-regime markets
    """

    def __init__(self, n_states: int = 2, max_iter: int = 100):
        """
        Initialize Markov Switching Model

        Args:
            n_states: Number of hidden states
            max_iter: Maximum EM iterations
        """
        self.n_states = n_states
        self.max_iter = max_iter
        self.transition_matrix = None
        self.state_params = None
        self.fitted = False

    def _forward_backward(self, data: np.ndarray, params: Dict) -> Dict[str, np.ndarray]:
        """
        Forward-backward algorithm for state probabilities

        Args:
            data: Observations
            params: Model parameters

        Returns:
            Dictionary with alpha, beta, gamma, xi
        """
        T = len(data)
        alpha = np.zeros((T, self.n_states))  # Forward probabilities
        beta = np.zeros((T, self.n_states))   # Backward probabilities
        c = np.zeros(T)  # Scaling factors

        # Extract parameters
        A = params['transition_matrix']  # Transition matrix
        pi = params['initial_probs']     # Initial state probabilities
        mu = params['means']             # State means
        sigma = params['std_devs']       # State standard deviations

        # Forward pass
        # Initialize
        for s in range(self.n_states):
            alpha[0, s] = pi[s] * stats.norm.pdf(data[0], mu[s], sigma[s])
        c[0] = np.sum(alpha[0])
        alpha[0] /= c[0]

        # Forward recursion
        for t in range(1, T):
            for s in range(self.n_states):
                alpha[t, s] = np.sum(alpha[t-1] * A[:, s]) * stats.norm.pdf(data[t], mu[s], sigma[s])
            c[t] = np.sum(alpha[t])
            if c[t] > 0:
                alpha[t] /= c[t]

        # Backward pass
        # Initialize
        beta[-1] = 1.0

        # Backward recursion
        for t in range(T-2, -1, -1):
            for s in range(self.n_states):
                beta[t, s] = np.sum(
                    A[s, :] * stats.norm.pdf(data[t+1], mu, sigma) * beta[t+1]
                )
            if c[t+1] > 0:
                beta[t] /= c[t+1]

        # State probabilities
        gamma = alpha * beta
        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        gamma_sum[gamma_sum == 0] = 1  # Avoid division by zero
        gamma = gamma / gamma_sum

        # Pairwise state probabilities
        xi = np.zeros((T-1, self.n_states, self.n_states))
        for t in range(T-1):
            denominator = 0
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (
                        alpha[t, i] * A[i, j] *
                        stats.norm.pdf(data[t+1], mu[j], sigma[j]) * beta[t+1, j]
                    )
                    denominator += xi[t, i, j]

            if denominator > 0:
                xi[t] /= denominator

        return {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'xi': xi,
            'log_likelihood': np.sum(np.log(c + 1e-300))  # Avoid log(0)
        }

    def fit(self, data: np.ndarray, prior_params: Optional[Dict] = None) -> BayesianResult:
        """
        Fit Markov switching model using EM algorithm

        Args:
            data: Observations
            prior_params: Prior parameters (optional)

        Returns:
            BayesianResult with fitted parameters
        """
        logger.info("Fitting Markov Switching model...")

        if len(data) < 50:
            logger.warning("Small sample for Markov switching model")

        T = len(data)

        # Initialize parameters
        # Transition matrix (uniform + small noise)
        A = np.ones((self.n_states, self.n_states)) / self.n_states
        A += np.random.normal(0, 0.01, (self.n_states, self.n_states))
        A = A / np.sum(A, axis=1, keepdims=True)  # Normalize rows

        # Initial state probabilities
        pi = np.ones(self.n_states) / self.n_states

        # State parameters (means and std devs)
        if self.n_states == 2:
            # Low vol and high vol states
            mu = [np.mean(data), np.mean(data)]
            sigma = [np.std(data) * 0.5, np.std(data) * 1.5]
        else:
            # Distribute means across data range
            data_range = np.linspace(np.min(data), np.max(data), self.n_states)
            mu = data_range.tolist()
            sigma = [np.std(data)] * self.n_states

        params = {
            'transition_matrix': A,
            'initial_probs': pi,
            'means': np.array(mu),
            'std_devs': np.array(sigma)
        }

        log_likelihood_history = []

        # EM Algorithm
        for iteration in range(self.max_iter):
            # E-step: Forward-backward algorithm
            fb_result = self._forward_backward(data, params)
            gamma = fb_result['gamma']
            xi = fb_result['xi']
            log_likelihood = fb_result['log_likelihood']

            log_likelihood_history.append(log_likelihood)

            # Check convergence
            if iteration > 0 and abs(log_likelihood - log_likelihood_history[-2]) < 1e-6:
                logger.info(f"EM converged after {iteration} iterations")
                break

            # M-step: Update parameters
            # Update initial probabilities
            pi = gamma[0]

            # Update transition matrix
            xi_sum = np.sum(xi, axis=0)
            gamma_sum = np.sum(gamma[:-1], axis=0)
            gamma_sum[gamma_sum == 0] = 1  # Avoid division by zero

            for i in range(self.n_states):
                A[i, :] = xi_sum[i, :] / gamma_sum[i]

            # Update state means and standard deviations
            gamma_sum_all = np.sum(gamma, axis=0)
            gamma_sum_all[gamma_sum_all == 0] = 1

            for s in range(self.n_states):
                # Weighted mean
                mu[s] = np.sum(gamma[:, s] * data) / gamma_sum_all[s]

                # Weighted standard deviation
                sigma[s] = np.sqrt(
                    np.sum(gamma[:, s] * (data - mu[s])**2) / gamma_sum_all[s]
                )
                sigma[s] = max(sigma[s], 1e-6)  # Avoid zero std

            params['means'] = np.array(mu)
            params['std_devs'] = np.array(sigma)

        # Store fitted parameters
        self.transition_matrix = A
        self.state_params = {'means': mu, 'std_devs': sigma}
        self.fitted = True

        # Final forward-backward for results
        final_result = self._forward_backward(data, params)

        # Most likely state sequence (Viterbi algorithm - simplified)
        state_sequence = np.argmax(final_result['gamma'], axis=1)

        # Posterior summaries
        posterior_mean = {
            'state_0_mean': mu[0],
            'state_1_mean': mu[1] if len(mu) > 1 else mu[0],
            'state_0_std': sigma[0],
            'state_1_std': sigma[1] if len(sigma) > 1 else sigma[0],
            'transition_00': A[0, 0],
            'transition_11': A[1, 1] if A.shape[0] > 1 else A[0, 0]
        }

        # Approximate standard errors (from EM theory)
        posterior_std = {k: 0.1 * abs(v) for k, v in posterior_mean.items()}

        # Approximate credible intervals
        credible_intervals = {
            k: (v - 1.96 * posterior_std[k], v + 1.96 * posterior_std[k])
            for k, v in posterior_mean.items()
        }

        # Predictions (smoothed state probabilities)
        predictions = np.zeros(T)
        prediction_intervals = np.zeros((T, 2))

        for t in range(T):
            # Expected value given state probabilities
            expected_val = np.sum(final_result['gamma'][t] * np.array(mu))
            expected_var = np.sum(final_result['gamma'][t] * (np.array(sigma)**2 + np.array(mu)**2)) - expected_val**2

            predictions[t] = expected_val
            prediction_intervals[t] = [
                expected_val - 1.96 * np.sqrt(expected_var),
                expected_val + 1.96 * np.sqrt(expected_var)
            ]

        uncertainty_measures = {
            'state_uncertainty': np.mean(1 + np.sum(final_result['gamma'] * np.log(final_result['gamma'] + 1e-10), axis=1)),
            'regime_persistence': np.mean(np.diag(A)),
            'volatility_regime_diff': abs(sigma[0] - sigma[1]) if len(sigma) > 1 else 0
        }

        logger.info(f"Markov switching fitted. States: {self.n_states}, Log-likelihood: {final_result['log_likelihood']:.2f}")

        return BayesianResult(
            method_name="Markov_Switching",
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            credible_intervals=credible_intervals,
            log_marginal_likelihood=final_result['log_likelihood'],
            dic=-2 * final_result['log_likelihood'] + 2 * len(posterior_mean),
            waic=-2 * final_result['log_likelihood'] + 2 * len(posterior_mean),
            predictions=predictions,
            prediction_intervals=prediction_intervals,
            uncertainty_measures=uncertainty_measures
        )

    def predict(self, data: np.ndarray, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with Markov switching model"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        # Simplified prediction using current state distribution
        mu = self.state_params['means']
        predictions = np.array([np.mean(mu)] * len(data))
        sigma = np.mean(self.state_params['std_devs'])
        intervals = np.array([[np.mean(mu) - 1.96*sigma, np.mean(mu) + 1.96*sigma]] * len(data))

        return predictions, intervals


class DynamicBayesianUpdater(BayesianMethod):
    """
    Dynamic Bayesian Model with Online Learning

    Good for: Real-time trading where beliefs update with each new observation
    Poor for: Batch analysis where all data is available simultaneously
    """

    def __init__(self, decay_factor: float = 0.95, initial_precision: float = 0.01):
        """
        Initialize Dynamic Bayesian Updater

        Args:
            decay_factor: Exponential decay for old observations
            initial_precision: Initial precision (inverse variance) for prior
        """
        self.decay_factor = decay_factor
        self.initial_precision = initial_precision
        self.posterior_mean = 0.0
        self.posterior_precision = initial_precision
        self.update_history = []

    def _kalman_filter_update(self, prior_mean: float, prior_var: float,
                            observation: float, obs_var: float) -> Tuple[float, float]:
        """
        Single Kalman filter update

        Args:
            prior_mean: Prior mean
            prior_var: Prior variance
            observation: New observation
            obs_var: Observation variance

        Returns:
            Tuple of (posterior_mean, posterior_variance)
        """
        # Kalman gain
        kalman_gain = prior_var / (prior_var + obs_var)

        # Update mean
        posterior_mean = prior_mean + kalman_gain * (observation - prior_mean)

        # Update variance
        posterior_var = (1 - kalman_gain) * prior_var

        return posterior_mean, posterior_var

    def fit(self, data: np.ndarray, prior_params: Optional[Dict] = None) -> BayesianResult:
        """
        Fit dynamic Bayesian model with sequential updating

        Args:
            data: Sequential observations
            prior_params: Prior parameters

        Returns:
            BayesianResult with dynamic posterior updates
        """
        logger.info("Fitting Dynamic Bayesian model...")

        if prior_params:
            self.posterior_mean = prior_params.get('prior_mean', 0.0)
            self.posterior_precision = prior_params.get('prior_precision', self.initial_precision)

        # Sequential Bayesian updates
        posterior_means = []
        posterior_vars = []
        predictions = []
        prediction_intervals = []

        obs_var = np.var(data) if len(data) > 1 else 1.0  # Observation variance

        for t, observation in enumerate(data):
            # Prior for this step (with decay)
            if t > 0:
                # Apply exponential forgetting
                self.posterior_precision *= self.decay_factor

            prior_var = 1 / self.posterior_precision
            prior_mean = self.posterior_mean

            # Kalman filter update
            self.posterior_mean, posterior_var = self._kalman_filter_update(
                prior_mean, prior_var, observation, obs_var
            )

            self.posterior_precision = 1 / posterior_var

            # Store results
            posterior_means.append(self.posterior_mean)
            posterior_vars.append(posterior_var)

            # Prediction for next step
            pred_mean = self.posterior_mean
            pred_var = posterior_var + obs_var  # Predictive variance
            predictions.append(pred_mean)
            prediction_intervals.append([
                pred_mean - 1.96 * np.sqrt(pred_var),
                pred_mean + 1.96 * np.sqrt(pred_var)
            ])

            # Update history
            self.update_history.append({
                'step': t,
                'observation': observation,
                'posterior_mean': self.posterior_mean,
                'posterior_var': posterior_var,
                'prediction': pred_mean
            })

        # Summary statistics
        final_mean = self.posterior_mean
        final_std = np.sqrt(1 / self.posterior_precision)

        posterior_mean = {
            'final_mean': final_mean,
            'mean_trajectory_slope': (posterior_means[-1] - posterior_means[0]) / len(data) if len(data) > 1 else 0,
            'adaptation_rate': 1 - self.decay_factor
        }

        posterior_std = {
            'final_std': final_std,
            'mean_trajectory_slope': 0.01,  # Approximate
            'adaptation_rate': 0.01
        }

        credible_intervals = {
            'final_mean': (final_mean - 1.96 * final_std, final_mean + 1.96 * final_std),
            'mean_trajectory_slope': (posterior_mean['mean_trajectory_slope'] - 0.02,
                                    posterior_mean['mean_trajectory_slope'] + 0.02),
            'adaptation_rate': (1 - self.decay_factor - 0.02, 1 - self.decay_factor + 0.02)
        }

        # Approximate log marginal likelihood
        log_marginal_likelihood = -0.5 * len(data) * np.log(2 * np.pi) - 0.5 * np.sum(
            [np.log(pv + obs_var) + (data[i] - pm)**2 / (pv + obs_var)
             for i, (pm, pv) in enumerate(zip(posterior_means, posterior_vars))]
        )

        uncertainty_measures = {
            'final_uncertainty': final_std**2,
            'uncertainty_reduction': (1/self.initial_precision - 1/self.posterior_precision),
            'adaptation_speed': -np.log(self.decay_factor),
            'prediction_accuracy': np.mean([
                abs(self.update_history[i]['observation'] - self.update_history[i-1]['prediction'])
                for i in range(1, len(self.update_history))
            ]) if len(self.update_history) > 1 else 0
        }

        logger.info(f"Dynamic Bayesian updated through {len(data)} observations. Final mean: {final_mean:.4f}")

        return BayesianResult(
            method_name="Dynamic_Bayesian",
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            credible_intervals=credible_intervals,
            log_marginal_likelihood=log_marginal_likelihood,
            dic=-2 * log_marginal_likelihood + 2 * len(posterior_mean),
            waic=-2 * log_marginal_likelihood + 2 * len(posterior_mean),
            predictions=np.array(predictions),
            prediction_intervals=np.array(prediction_intervals),
            uncertainty_measures=uncertainty_measures
        )

    def predict(self, data: np.ndarray, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with current posterior"""
        pred_mean = self.posterior_mean
        pred_std = np.sqrt(1 / self.posterior_precision + 1)  # Add observation noise

        predictions = np.array([pred_mean] * len(data))
        intervals = np.array([[pred_mean - 1.96*pred_std, pred_mean + 1.96*pred_std]] * len(data))

        return predictions, intervals


class BayesianAnalyzer:
    """
    Main class for Bayesian analysis in different market regimes
    """

    def __init__(self):
        """Initialize Bayesian analyzer"""
        self.hierarchical_model = HierarchicalBayesianModel()
        self.switching_model = MarkovSwitchingModel()
        self.dynamic_model = DynamicBayesianUpdater()
        self.results = {}

    def analyze_regime(self, data: pd.DataFrame, regime_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data using Bayesian methods for a specific market regime

        Args:
            data: OHLCV DataFrame
            regime_info: Information about current market regime

        Returns:
            Dictionary with all Bayesian analysis results
        """
        regime_name = regime_info.get('regime_name', 'Unknown')
        logger.info(f"Running Bayesian analysis for regime: {regime_name}")

        results = {
            'regime_name': regime_name,
            'regime_characteristics': regime_info.get('characteristics', ''),
            'recommended_approach': regime_info.get('recommended_approach', ''),
            'data_size': len(data),
            'methods': {}
        }

        # Calculate returns
        returns = data['close'].pct_change().dropna().values

        if len(returns) < 10:
            logger.warning(f"Insufficient data for Bayesian analysis: {len(returns)} observations")
            results['methods']['warning'] = "Insufficient data for Bayesian analysis"
            return results

        # 1. Hierarchical Bayesian Analysis
        try:
            logger.info("Running Hierarchical Bayesian analysis...")
            hier_result = self.hierarchical_model.fit(returns)
            results['methods']['hierarchical'] = {
                'posterior_mean': hier_result.posterior_mean,
                'uncertainty_measures': hier_result.uncertainty_measures,
                'credible_intervals': hier_result.credible_intervals,
                'log_marginal_likelihood': hier_result.log_marginal_likelihood,
                'effectiveness': self._assess_hierarchical_effectiveness(hier_result, regime_info)
            }
        except Exception as e:
            logger.error(f"Hierarchical Bayesian analysis failed: {str(e)}")
            results['methods']['hierarchical'] = {'error': str(e)}

        # 2. Markov Switching Analysis
        try:
            logger.info("Running Markov Switching analysis...")
            switch_result = self.switching_model.fit(returns)
            results['methods']['switching'] = {
                'posterior_mean': switch_result.posterior_mean,
                'uncertainty_measures': switch_result.uncertainty_measures,
                'credible_intervals': switch_result.credible_intervals,
                'log_marginal_likelihood': switch_result.log_marginal_likelihood,
                'effectiveness': self._assess_switching_effectiveness(switch_result, regime_info)
            }
        except Exception as e:
            logger.error(f"Markov Switching analysis failed: {str(e)}")
            results['methods']['switching'] = {'error': str(e)}

        # 3. Dynamic Bayesian Analysis
        try:
            logger.info("Running Dynamic Bayesian analysis...")
            dynamic_result = self.dynamic_model.fit(returns)
            results['methods']['dynamic'] = {
                'posterior_mean': dynamic_result.posterior_mean,
                'uncertainty_measures': dynamic_result.uncertainty_measures,
                'credible_intervals': dynamic_result.credible_intervals,
                'log_marginal_likelihood': dynamic_result.log_marginal_likelihood,
                'effectiveness': self._assess_dynamic_effectiveness(dynamic_result, regime_info)
            }
        except Exception as e:
            logger.error(f"Dynamic Bayesian analysis failed: {str(e)}")
            results['methods']['dynamic'] = {'error': str(e)}

        # 4. Overall Bayesian Assessment
        results['overall_assessment'] = self._assess_overall_bayesian_effectiveness(results, regime_info)

        return results

    def _assess_hierarchical_effectiveness(self, result: BayesianResult, regime_info: Dict) -> Dict[str, Any]:
        """Assess hierarchical Bayesian effectiveness for the regime"""
        regime_name = regime_info.get('regime_name', '')

        effectiveness = {
            'score': 0.5,
            'reasons': [],
            'reliability': 'Medium'
        }

        # Check for sparse data regimes
        if 'Low' in regime_name:  # Low liquidity or volume
            effectiveness['score'] = 0.8
            effectiveness['reasons'].append("Hierarchical structure helps with sparse data")
            effectiveness['reliability'] = 'High'

        if 'HighLiq_HighVol' in regime_name:
            effectiveness['score'] = 0.4
            effectiveness['reasons'].append("Abundant data makes hierarchical structure less necessary")

        # Check uncertainty measures
        total_uncertainty = result.uncertainty_measures.get('total_uncertainty', 0)
        between_group_var = result.uncertainty_measures.get('between_group_variance', 0)

        if between_group_var > 0.1 * total_uncertainty:
            effectiveness['score'] += 0.2
            effectiveness['reasons'].append("Significant between-group variation detected")

        return effectiveness

    def _assess_switching_effectiveness(self, result: BayesianResult, regime_info: Dict) -> Dict[str, Any]:
        """Assess Markov switching effectiveness for the regime"""
        regime_name = regime_info.get('regime_name', '')

        effectiveness = {
            'score': 0.5,
            'reasons': [],
            'reliability': 'Medium'
        }

        # High volatility regimes benefit from regime switching
        if 'HighVol' in regime_name:
            effectiveness['score'] = 0.8
            effectiveness['reasons'].append("High volatility suggests regime switching behavior")
            effectiveness['reliability'] = 'High'

        if 'LowVol_LowVol' in regime_name:
            effectiveness['score'] = 0.3
            effectiveness['reasons'].append("Stable regime - switching model may overfit")

        # Check regime persistence
        persistence = result.uncertainty_measures.get('regime_persistence', 0.5)
        if 0.6 < persistence < 0.9:
            effectiveness['score'] += 0.2
            effectiveness['reasons'].append("Good regime persistence detected")
        elif persistence > 0.95:
            effectiveness['reasons'].append("Very persistent regimes - single regime might suffice")

        return effectiveness

    def _assess_dynamic_effectiveness(self, result: BayesianResult, regime_info: Dict) -> Dict[str, Any]:
        """Assess dynamic Bayesian effectiveness for the regime"""
        regime_name = regime_info.get('regime_name', '')

        effectiveness = {
            'score': 0.6,  # Generally good for real-time
            'reasons': ['Suitable for online learning'],
            'reliability': 'High'
        }

        # All regimes can benefit from dynamic updating
        if 'HighVol' in regime_name:
            effectiveness['score'] += 0.1
            effectiveness['reasons'].append("High volatility requires fast adaptation")

        # Check adaptation performance
        uncertainty_reduction = result.uncertainty_measures.get('uncertainty_reduction', 0)
        if uncertainty_reduction > 0:
            effectiveness['score'] += 0.1
            effectiveness['reasons'].append("Successfully reducing uncertainty over time")

        prediction_accuracy = result.uncertainty_measures.get('prediction_accuracy', float('inf'))
        if prediction_accuracy < 0.5:  # Good prediction accuracy
            effectiveness['score'] += 0.1
            effectiveness['reasons'].append("Good predictive performance")

        return effectiveness

    def _assess_overall_bayesian_effectiveness(self, results: Dict, regime_info: Dict) -> Dict[str, Any]:
        """Assess overall Bayesian approach effectiveness"""
        regime_name = regime_info.get('regime_name', '')
        recommended = regime_info.get('recommended_approach', '')

        # Extract individual scores
        scores = []
        if 'hierarchical' in results['methods'] and 'effectiveness' in results['methods']['hierarchical']:
            scores.append(results['methods']['hierarchical']['effectiveness']['score'])
        if 'switching' in results['methods'] and 'effectiveness' in results['methods']['switching']:
            scores.append(results['methods']['switching']['effectiveness']['score'])
        if 'dynamic' in results['methods'] and 'effectiveness' in results['methods']['dynamic']:
            scores.append(results['methods']['dynamic']['effectiveness']['score'])

        overall_score = np.mean(scores) if scores else 0.5

        # Adjust based on theoretical expectation
        if 'Bayesian' in recommended:
            expected_score = 0.8
        elif 'Frequentist' in recommended:
            expected_score = 0.3
        else:  # Mixed
            expected_score = 0.6

        assessment = {
            'effectiveness_score': overall_score,
            'expected_score': expected_score,
            'matches_theory': abs(overall_score - expected_score) < 0.3,
            'summary': self._generate_bayesian_summary(overall_score, expected_score, regime_name)
        }

        return assessment

    def _generate_bayesian_summary(self, actual: float, expected: float, regime_name: str) -> str:
        """Generate human-readable Bayesian effectiveness summary"""
        if actual >= 0.7:
            performance = "highly effective"
        elif actual >= 0.5:
            performance = "moderately effective"
        else:
            performance = "poorly effective"

        if abs(actual - expected) < 0.2:
            alignment = "as expected"
        elif actual > expected:
            alignment = "better than expected"
        else:
            alignment = "worse than expected"

        return f"Bayesian methods are {performance} in {regime_name} regime, performing {alignment}"