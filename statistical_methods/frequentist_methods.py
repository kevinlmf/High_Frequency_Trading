"""
Frequentist Statistical Methods for HFT
======================================

Implementation of classical frequentist approaches for HFT trading:
1. GARCH models for volatility modeling
2. Hawkes processes for market microstructure
3. Classical statistical tests and signal detection
4. Maximum likelihood estimation methods

These methods work well in high-volume, liquid markets with sufficient data.
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
class FrequentistResult:
    """Result container for frequentist methods"""
    method_name: str
    parameters: Dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    predictions: np.ndarray
    standard_errors: np.ndarray


class FrequentistMethod(ABC):
    """Abstract base class for frequentist methods"""

    @abstractmethod
    def fit(self, data: np.ndarray) -> FrequentistResult:
        """Fit the model to data"""
        pass

    @abstractmethod
    def predict(self, data: np.ndarray, steps: int = 1) -> np.ndarray:
        """Make predictions"""
        pass


class GARCHModel(FrequentistMethod):
    """
    GARCH(1,1) Model for Volatility Prediction

    Good for: High liquidity + High volume markets where volatility clustering is evident
    Poor for: Sparse, low-volume markets with insufficient data for MLE convergence
    """

    def __init__(self, model_type: str = "garch"):
        """
        Initialize GARCH model

        Args:
            model_type: "garch", "egarch", or "gjr-garch"
        """
        self.model_type = model_type
        self.params = None
        self.fitted_data = None

    def _garch_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Negative log-likelihood for GARCH(1,1)

        Args:
            params: [omega, alpha, beta] parameters
            returns: Return series

        Returns:
            Negative log-likelihood value
        """
        omega, alpha, beta = params
        T = len(returns)

        # Initialize
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(returns)  # Initial variance

        # GARCH recursion: σ²ₜ = ω + α⋅ε²ₜ₋₁ + β⋅σ²ₜ₋₁
        for t in range(1, T):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]

        # Avoid numerical issues
        sigma2 = np.maximum(sigma2, 1e-10)

        # Log-likelihood
        log_likelihood = -0.5 * np.sum(
            np.log(2 * np.pi * sigma2) + returns**2 / sigma2
        )

        return -log_likelihood  # Return negative for minimization

    def fit(self, returns: np.ndarray) -> FrequentistResult:
        """
        Fit GARCH model using Maximum Likelihood Estimation

        Args:
            returns: Return series

        Returns:
            FrequentistResult with fitted parameters and statistics
        """
        logger.info(f"Fitting {self.model_type.upper()} model...")

        # Remove any NaN values
        returns = returns[~np.isnan(returns)]

        if len(returns) < 50:
            logger.warning("Insufficient data for GARCH estimation (< 50 observations)")
            # Return dummy result for sparse data
            return FrequentistResult(
                method_name=self.model_type.upper(),
                parameters={'omega': 0.0, 'alpha': 0.0, 'beta': 0.9},
                log_likelihood=-np.inf,
                aic=np.inf,
                bic=np.inf,
                p_values={'omega': 1.0, 'alpha': 1.0, 'beta': 1.0},
                confidence_intervals={'omega': (-1, 1), 'alpha': (-1, 1), 'beta': (-1, 1)},
                predictions=np.array([np.var(returns)] * len(returns)),
                standard_errors=np.array([np.std(returns)] * len(returns))
            )

        # Initial parameter estimates
        unconditional_var = np.var(returns)
        initial_params = np.array([
            unconditional_var * 0.01,  # omega
            0.1,                       # alpha
            0.85                       # beta
        ])

        # Parameter bounds for stability: α + β < 1
        bounds = [
            (1e-8, unconditional_var),  # omega > 0
            (1e-8, 1.0),               # alpha > 0
            (1e-8, 0.99)               # beta > 0
        ]

        # Constraints: alpha + beta < 1 for stationarity
        constraints = {'type': 'ineq', 'fun': lambda x: 0.99 - x[1] - x[2]}

        try:
            # Optimize parameters
            result = minimize(
                self._garch_likelihood,
                initial_params,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

            if not result.success:
                logger.warning(f"GARCH optimization failed: {result.message}")

            # Extract parameters
            omega, alpha, beta = result.x
            self.params = {'omega': omega, 'alpha': alpha, 'beta': beta}

            # Calculate information criteria
            log_likelihood = -result.fun
            n_params = 3
            n_obs = len(returns)
            aic = -2 * log_likelihood + 2 * n_params
            bic = -2 * log_likelihood + np.log(n_obs) * n_params

            # Calculate standard errors (using Hessian approximation)
            try:
                # Simple numerical Hessian approximation
                eps = 1e-5
                hessian = np.zeros((3, 3))
                for i in range(3):
                    for j in range(3):
                        param_plus = result.x.copy()
                        param_minus = result.x.copy()
                        param_plus[i] += eps
                        param_plus[j] += eps
                        param_minus[i] -= eps
                        param_minus[j] -= eps

                        hessian[i, j] = (
                            self._garch_likelihood(param_plus, returns) -
                            self._garch_likelihood(param_minus, returns)
                        ) / (4 * eps * eps)

                # Standard errors from inverse Hessian
                try:
                    inv_hessian = np.linalg.inv(hessian)
                    std_errors = np.sqrt(np.diag(inv_hessian))
                except np.linalg.LinAlgError:
                    std_errors = np.array([0.1, 0.1, 0.1])  # Fallback

            except:
                std_errors = np.array([0.1, 0.1, 0.1])  # Fallback

            # Calculate p-values (Wald test: parameter / std_error ~ N(0,1))
            t_stats = result.x / std_errors
            p_values = {
                'omega': 2 * (1 - stats.norm.cdf(np.abs(t_stats[0]))),
                'alpha': 2 * (1 - stats.norm.cdf(np.abs(t_stats[1]))),
                'beta': 2 * (1 - stats.norm.cdf(np.abs(t_stats[2])))
            }

            # 95% confidence intervals
            ci_multiplier = stats.norm.ppf(0.975)  # 1.96
            confidence_intervals = {
                'omega': (omega - ci_multiplier * std_errors[0], omega + ci_multiplier * std_errors[0]),
                'alpha': (alpha - ci_multiplier * std_errors[1], alpha + ci_multiplier * std_errors[1]),
                'beta': (beta - ci_multiplier * std_errors[2], beta + ci_multiplier * std_errors[2])
            }

            # Generate fitted volatilities
            T = len(returns)
            sigma2 = np.zeros(T)
            sigma2[0] = unconditional_var

            for t in range(1, T):
                sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]

            predictions = np.sqrt(sigma2)
            self.fitted_data = returns

            logger.info(f"GARCH fitted successfully: α={alpha:.4f}, β={beta:.4f}, α+β={alpha+beta:.4f}")

            return FrequentistResult(
                method_name=self.model_type.upper(),
                parameters=self.params,
                log_likelihood=log_likelihood,
                aic=aic,
                bic=bic,
                p_values=p_values,
                confidence_intervals=confidence_intervals,
                predictions=predictions,
                standard_errors=std_errors
            )

        except Exception as e:
            logger.error(f"GARCH fitting failed: {str(e)}")
            # Return dummy result on failure
            return FrequentistResult(
                method_name=self.model_type.upper(),
                parameters={'omega': 0.0, 'alpha': 0.0, 'beta': 0.9},
                log_likelihood=-np.inf,
                aic=np.inf,
                bic=np.inf,
                p_values={'omega': 1.0, 'alpha': 1.0, 'beta': 1.0},
                confidence_intervals={'omega': (-1, 1), 'alpha': (-1, 1), 'beta': (-1, 1)},
                predictions=np.array([np.std(returns)] * len(returns)),
                standard_errors=np.array([0.1, 0.1, 0.1])
            )

    def predict(self, data: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Predict future volatilities

        Args:
            data: Historical returns
            steps: Number of steps to forecast

        Returns:
            Array of predicted volatilities
        """
        if self.params is None:
            raise ValueError("Model must be fitted before prediction")

        omega, alpha, beta = self.params['omega'], self.params['alpha'], self.params['beta']

        # Current volatility (from last observation)
        last_return = data[-1] if len(data) > 0 else 0
        current_vol2 = omega + alpha * last_return**2 + beta * np.var(data)

        # Forecast future volatilities
        forecasts = []
        vol2 = current_vol2

        for _ in range(steps):
            # For multi-step ahead: E[ε²ₜ₊ₕ] = E[σ²ₜ₊ₕ]
            vol2 = omega + (alpha + beta) * vol2
            forecasts.append(np.sqrt(vol2))

        return np.array(forecasts)


class HawkesProcess(FrequentistMethod):
    """
    Hawkes Process for Modeling Event Clustering

    Good for: High-frequency data with clear event clustering (market orders, trades)
    Poor for: Low-frequency data or markets without self-exciting behavior
    """

    def __init__(self, kernel: str = "exponential"):
        """
        Initialize Hawkes process

        Args:
            kernel: Type of decay kernel ("exponential", "power_law")
        """
        self.kernel = kernel
        self.params = None

    def _exponential_kernel(self, t: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """Exponential decay kernel"""
        return alpha * beta * np.exp(-beta * t)

    def _hawkes_likelihood(self, params: np.ndarray, event_times: np.ndarray, T: float) -> float:
        """
        Negative log-likelihood for exponential Hawkes process

        Args:
            params: [mu, alpha, beta] - baseline intensity, self-excitation, decay
            event_times: Array of event times
            T: Total observation period

        Returns:
            Negative log-likelihood
        """
        mu, alpha, beta = params

        if mu <= 0 or alpha < 0 or beta <= 0:
            return 1e10  # Invalid parameters

        n_events = len(event_times)

        # Compensator (integral of intensity)
        compensator = mu * T

        # Self-excitation component
        for i, ti in enumerate(event_times):
            # Contribution from events before ti
            prior_events = event_times[event_times < ti]
            if len(prior_events) > 0:
                excitation = np.sum(alpha * (1 - np.exp(-beta * (ti - prior_events))))
                compensator += excitation

        # Log-likelihood of events
        log_intensity_sum = 0
        for i, ti in enumerate(event_times):
            # Baseline intensity
            intensity = mu

            # Self-excitation from previous events
            prior_events = event_times[event_times < ti]
            if len(prior_events) > 0:
                intensity += np.sum(alpha * np.exp(-beta * (ti - prior_events)))

            if intensity <= 0:
                return 1e10  # Invalid intensity

            log_intensity_sum += np.log(intensity)

        log_likelihood = log_intensity_sum - compensator
        return -log_likelihood  # Return negative for minimization

    def fit(self, event_times: np.ndarray, T: Optional[float] = None) -> FrequentistResult:
        """
        Fit Hawkes process using MLE

        Args:
            event_times: Array of event occurrence times
            T: Total observation period (if None, use max(event_times))

        Returns:
            FrequentistResult with fitted parameters
        """
        logger.info("Fitting Hawkes process...")

        if T is None:
            T = event_times.max() if len(event_times) > 0 else 1.0

        if len(event_times) < 10:
            logger.warning("Insufficient events for Hawkes estimation (< 10 events)")
            return FrequentistResult(
                method_name="Hawkes",
                parameters={'mu': 0.1, 'alpha': 0.0, 'beta': 1.0},
                log_likelihood=-np.inf,
                aic=np.inf,
                bic=np.inf,
                p_values={'mu': 1.0, 'alpha': 1.0, 'beta': 1.0},
                confidence_intervals={'mu': (0, 1), 'alpha': (0, 1), 'beta': (0, 2)},
                predictions=np.array([0.1] * len(event_times)),
                standard_errors=np.array([0.1, 0.1, 0.1])
            )

        # Initial parameter estimates
        n_events = len(event_times)
        initial_mu = n_events / T  # Average rate
        initial_params = np.array([initial_mu, 0.5, 1.0])

        # Parameter bounds
        bounds = [
            (1e-6, 10 * initial_mu),  # mu > 0
            (0, 0.99),                # 0 ≤ alpha < 1 (for stability)
            (1e-6, 100)               # beta > 0
        ]

        try:
            # Optimize parameters
            result = minimize(
                self._hawkes_likelihood,
                initial_params,
                args=(event_times, T),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )

            if not result.success:
                logger.warning(f"Hawkes optimization failed: {result.message}")

            # Extract parameters
            mu, alpha, beta = result.x
            self.params = {'mu': mu, 'alpha': alpha, 'beta': beta}

            # Calculate information criteria
            log_likelihood = -result.fun
            n_params = 3
            aic = -2 * log_likelihood + 2 * n_params
            bic = -2 * log_likelihood + np.log(n_events) * n_params

            # Simple standard error approximation
            std_errors = np.array([0.1 * mu, 0.1 * alpha, 0.1 * beta])

            # P-values (approximate)
            p_values = {
                'mu': 0.01 if mu > 0.1 * initial_mu else 0.5,
                'alpha': 0.01 if alpha > 0.1 else 0.5,
                'beta': 0.01 if beta > 0.1 else 0.5
            }

            # Confidence intervals (approximate)
            confidence_intervals = {
                'mu': (mu - 1.96 * std_errors[0], mu + 1.96 * std_errors[0]),
                'alpha': (max(0, alpha - 1.96 * std_errors[1]), min(0.99, alpha + 1.96 * std_errors[1])),
                'beta': (max(0.01, beta - 1.96 * std_errors[2]), beta + 1.96 * std_errors[2])
            }

            # Generate intensity predictions at event times
            intensities = np.zeros(len(event_times))
            for i, ti in enumerate(event_times):
                intensity = mu
                prior_events = event_times[event_times < ti]
                if len(prior_events) > 0:
                    intensity += np.sum(alpha * np.exp(-beta * (ti - prior_events)))
                intensities[i] = intensity

            logger.info(f"Hawkes fitted: μ={mu:.4f}, α={alpha:.4f}, β={beta:.4f}")

            return FrequentistResult(
                method_name="Hawkes",
                parameters=self.params,
                log_likelihood=log_likelihood,
                aic=aic,
                bic=bic,
                p_values=p_values,
                confidence_intervals=confidence_intervals,
                predictions=intensities,
                standard_errors=std_errors
            )

        except Exception as e:
            logger.error(f"Hawkes fitting failed: {str(e)}")
            return FrequentistResult(
                method_name="Hawkes",
                parameters={'mu': initial_mu, 'alpha': 0.0, 'beta': 1.0},
                log_likelihood=-np.inf,
                aic=np.inf,
                bic=np.inf,
                p_values={'mu': 1.0, 'alpha': 1.0, 'beta': 1.0},
                confidence_intervals={'mu': (0, 1), 'alpha': (0, 1), 'beta': (0, 2)},
                predictions=np.array([initial_mu] * len(event_times)),
                standard_errors=np.array([0.1, 0.1, 0.1])
            )

    def predict(self, data: np.ndarray, steps: int = 1) -> np.ndarray:
        """Predict future intensity (simplified)"""
        if self.params is None:
            raise ValueError("Model must be fitted before prediction")

        mu = self.params['mu']
        return np.array([mu] * steps)  # Simplified: return baseline intensity


class ClassicalTests:
    """
    Classical Statistical Tests for Trading Signals

    Good for: Large samples with approximately normal distributions
    Poor for: Small samples, non-normal distributions, regime changes
    """

    @staticmethod
    def augmented_dickey_fuller_test(series: np.ndarray, lags: int = 1) -> Dict[str, Any]:
        """
        Augmented Dickey-Fuller test for unit root (stationarity)

        Args:
            series: Time series data
            lags: Number of lags to include

        Returns:
            Dictionary with test results
        """
        from scipy import stats

        n = len(series)
        if n < 20:
            return {
                'statistic': np.nan,
                'p_value': 1.0,
                'critical_values': {'1%': np.nan, '5%': np.nan, '10%': np.nan},
                'is_stationary': False,
                'method': 'ADF_insufficient_data'
            }

        # Simplified ADF test implementation
        y = series[1:]
        x1 = series[:-1]

        # Add lagged differences if lags > 1
        X = x1.reshape(-1, 1)
        if lags > 1:
            for i in range(1, lags):
                if len(series) > i + 1:
                    lagged_diff = np.diff(series[:-i])
                    if len(lagged_diff) == len(y):
                        X = np.column_stack([X, lagged_diff])

        # OLS regression: Δy = α + βy₋₁ + εₜ
        try:
            # Add constant
            X = np.column_stack([np.ones(len(X)), X])

            # OLS estimation
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta
            sigma2 = np.sum(residuals**2) / (len(residuals) - len(beta))

            # Standard error of β (coefficient on y₋₁)
            try:
                se_beta = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
                t_stat = beta[1] / se_beta  # Test H₀: β = 0 (unit root)
            except:
                t_stat = np.nan
                se_beta = np.nan

            # Critical values (approximate)
            critical_values = {'1%': -3.43, '5%': -2.86, '10%': -2.57}

            # P-value approximation (rough)
            if not np.isnan(t_stat):
                p_value = stats.norm.cdf(t_stat) if t_stat < 0 else 1 - stats.norm.cdf(t_stat)
            else:
                p_value = 1.0

            is_stationary = t_stat < critical_values['5%'] and not np.isnan(t_stat)

            return {
                'statistic': t_stat,
                'p_value': p_value,
                'critical_values': critical_values,
                'is_stationary': is_stationary,
                'method': 'ADF'
            }

        except Exception as e:
            return {
                'statistic': np.nan,
                'p_value': 1.0,
                'critical_values': {'1%': np.nan, '5%': np.nan, '10%': np.nan},
                'is_stationary': False,
                'method': f'ADF_failed: {str(e)}'
            }

    @staticmethod
    def ljung_box_test(residuals: np.ndarray, lags: int = 10) -> Dict[str, Any]:
        """
        Ljung-Box test for autocorrelation in residuals

        Args:
            residuals: Model residuals
            lags: Number of lags to test

        Returns:
            Dictionary with test results
        """
        n = len(residuals)
        if n < max(lags + 5, 20):
            return {
                'statistic': np.nan,
                'p_value': 1.0,
                'has_autocorr': False,
                'method': 'LB_insufficient_data'
            }

        # Calculate sample autocorrelations
        try:
            mean_resid = np.mean(residuals)
            centered_resid = residuals - mean_resid
            c0 = np.mean(centered_resid**2)  # Variance

            autocorrs = []
            for k in range(1, lags + 1):
                if n > k:
                    ck = np.mean(centered_resid[:-k] * centered_resid[k:])
                    rk = ck / c0 if c0 != 0 else 0
                    autocorrs.append(rk)
                else:
                    autocorrs.append(0)

            autocorrs = np.array(autocorrs)

            # Ljung-Box statistic
            Q = n * (n + 2) * np.sum(autocorrs**2 / np.arange(n-1, n-lags-1, -1))

            # P-value (chi-squared distribution with 'lags' degrees of freedom)
            p_value = 1 - stats.chi2.cdf(Q, lags)

            has_autocorr = p_value < 0.05  # Reject null of no autocorrelation

            return {
                'statistic': Q,
                'p_value': p_value,
                'has_autocorr': has_autocorr,
                'method': 'Ljung-Box'
            }

        except Exception as e:
            return {
                'statistic': np.nan,
                'p_value': 1.0,
                'has_autocorr': False,
                'method': f'LB_failed: {str(e)}'
            }

    @staticmethod
    def jarque_bera_test(data: np.ndarray) -> Dict[str, Any]:
        """
        Jarque-Bera test for normality

        Args:
            data: Data to test

        Returns:
            Dictionary with test results
        """
        n = len(data)
        if n < 8:
            return {
                'statistic': np.nan,
                'p_value': 1.0,
                'is_normal': False,
                'method': 'JB_insufficient_data'
            }

        try:
            # Calculate skewness and kurtosis
            mean_data = np.mean(data)
            std_data = np.std(data, ddof=1)
            standardized = (data - mean_data) / std_data

            skewness = np.mean(standardized**3)
            kurtosis = np.mean(standardized**4) - 3  # Excess kurtosis

            # Jarque-Bera statistic
            JB = n * (skewness**2 / 6 + kurtosis**2 / 24)

            # P-value (chi-squared distribution with 2 degrees of freedom)
            p_value = 1 - stats.chi2.cdf(JB, 2)

            is_normal = p_value > 0.05  # Fail to reject null of normality

            return {
                'statistic': JB,
                'p_value': p_value,
                'is_normal': is_normal,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'method': 'Jarque-Bera'
            }

        except Exception as e:
            return {
                'statistic': np.nan,
                'p_value': 1.0,
                'is_normal': False,
                'method': f'JB_failed: {str(e)}'
            }


class FrequentistAnalyzer:
    """
    Main class for frequentist analysis in different market regimes
    """

    def __init__(self):
        """Initialize frequentist analyzer"""
        self.garch_model = GARCHModel()
        self.hawkes_model = HawkesProcess()
        self.tests = ClassicalTests()
        self.results = {}

    def analyze_regime(self, data: pd.DataFrame, regime_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data using frequentist methods for a specific market regime

        Args:
            data: OHLCV DataFrame
            regime_info: Information about current market regime

        Returns:
            Dictionary with all frequentist analysis results
        """
        regime_name = regime_info.get('regime_name', 'Unknown')
        logger.info(f"Running frequentist analysis for regime: {regime_name}")

        results = {
            'regime_name': regime_name,
            'regime_characteristics': regime_info.get('characteristics', ''),
            'recommended_approach': regime_info.get('recommended_approach', ''),
            'data_size': len(data),
            'methods': {}
        }

        # Calculate returns
        returns = data['close'].pct_change().dropna().values

        if len(returns) < 20:
            logger.warning(f"Insufficient data for frequentist analysis: {len(returns)} observations")
            results['methods']['warning'] = "Insufficient data for reliable frequentist analysis"
            return results

        # 1. GARCH Analysis
        try:
            logger.info("Running GARCH analysis...")
            garch_result = self.garch_model.fit(returns)
            results['methods']['garch'] = {
                'parameters': garch_result.parameters,
                'log_likelihood': garch_result.log_likelihood,
                'aic': garch_result.aic,
                'bic': garch_result.bic,
                'p_values': garch_result.p_values,
                'persistence': (
                    garch_result.parameters.get('alpha', 0) +
                    garch_result.parameters.get('beta', 0)
                ),
                'effectiveness': self._assess_garch_effectiveness(garch_result, regime_info)
            }
        except Exception as e:
            logger.error(f"GARCH analysis failed: {str(e)}")
            results['methods']['garch'] = {'error': str(e)}

        # 2. Hawkes Process Analysis (using volume spikes as events)
        try:
            logger.info("Running Hawkes process analysis...")
            # Create event times from volume spikes
            volume_threshold = data['volume'].quantile(0.8)
            volume_events = data[data['volume'] > volume_threshold].index
            event_times = np.array([(t - data.index[0]).total_seconds() / 3600
                                  for t in volume_events])  # Hours since start

            if len(event_times) > 5:
                hawkes_result = self.hawkes_model.fit(event_times)
                results['methods']['hawkes'] = {
                    'parameters': hawkes_result.parameters,
                    'log_likelihood': hawkes_result.log_likelihood,
                    'aic': hawkes_result.aic,
                    'n_events': len(event_times),
                    'self_excitation': hawkes_result.parameters.get('alpha', 0),
                    'effectiveness': self._assess_hawkes_effectiveness(hawkes_result, regime_info)
                }
            else:
                results['methods']['hawkes'] = {'warning': 'Too few volume events for Hawkes analysis'}

        except Exception as e:
            logger.error(f"Hawkes analysis failed: {str(e)}")
            results['methods']['hawkes'] = {'error': str(e)}

        # 3. Classical Statistical Tests
        try:
            logger.info("Running classical statistical tests...")

            # Stationarity test
            adf_result = self.tests.augmented_dickey_fuller_test(returns)

            # Autocorrelation test on returns
            lb_returns = self.tests.ljung_box_test(returns)

            # Normality test
            jb_result = self.tests.jarque_bera_test(returns)

            results['methods']['statistical_tests'] = {
                'stationarity': adf_result,
                'autocorrelation': lb_returns,
                'normality': jb_result,
                'effectiveness': self._assess_tests_effectiveness(
                    adf_result, lb_returns, jb_result, regime_info
                )
            }

        except Exception as e:
            logger.error(f"Statistical tests failed: {str(e)}")
            results['methods']['statistical_tests'] = {'error': str(e)}

        # 4. Overall Frequentist Assessment
        results['overall_assessment'] = self._assess_overall_effectiveness(results, regime_info)

        return results

    def _assess_garch_effectiveness(self, garch_result: FrequentistResult, regime_info: Dict) -> Dict[str, Any]:
        """Assess GARCH model effectiveness for the given regime"""
        regime_name = regime_info.get('regime_name', '')

        effectiveness = {
            'score': 0.5,  # Default neutral
            'reasons': [],
            'reliability': 'Medium'
        }

        # Check if we have valid results
        if garch_result.log_likelihood == -np.inf:
            effectiveness['score'] = 0.1
            effectiveness['reasons'].append("Model failed to converge")
            effectiveness['reliability'] = 'Low'
            return effectiveness

        # Assess based on regime characteristics
        if 'HighLiq_HighVol' in regime_name:
            effectiveness['score'] = 0.8
            effectiveness['reasons'].append("High liquidity + volume: sufficient data for MLE")
            effectiveness['reliability'] = 'High'
        elif 'LowVol_LowVol' in regime_name:
            effectiveness['score'] = 0.3
            effectiveness['reasons'].append("Low volume: insufficient data for stable estimation")
            effectiveness['reliability'] = 'Low'
        elif 'HighVol' in regime_name:  # High volatility
            effectiveness['score'] = 0.7
            effectiveness['reasons'].append("High volatility: GARCH captures clustering well")

        # Check parameter significance
        alpha = garch_result.parameters.get('alpha', 0)
        beta = garch_result.parameters.get('beta', 0)

        if garch_result.p_values.get('alpha', 1) < 0.05:
            effectiveness['score'] += 0.1
            effectiveness['reasons'].append("Alpha parameter statistically significant")

        if alpha + beta > 0.99:
            effectiveness['score'] -= 0.2
            effectiveness['reasons'].append("Parameters near unit root - unstable")

        return effectiveness

    def _assess_hawkes_effectiveness(self, hawkes_result: FrequentistResult, regime_info: Dict) -> Dict[str, Any]:
        """Assess Hawkes process effectiveness for the given regime"""
        regime_name = regime_info.get('regime_name', '')

        effectiveness = {
            'score': 0.5,
            'reasons': [],
            'reliability': 'Medium'
        }

        if hawkes_result.log_likelihood == -np.inf:
            effectiveness['score'] = 0.1
            effectiveness['reasons'].append("Model failed to converge")
            effectiveness['reliability'] = 'Low'
            return effectiveness

        # Assess based on regime
        if 'HighVol' in regime_name and 'High' in regime_name:  # High volume
            effectiveness['score'] = 0.8
            effectiveness['reasons'].append("High volume: plenty of events for Hawkes estimation")
            effectiveness['reliability'] = 'High'
        elif 'LowVol' in regime_name:
            effectiveness['score'] = 0.3
            effectiveness['reasons'].append("Low volume: few events, unreliable clustering detection")
            effectiveness['reliability'] = 'Low'

        # Check self-excitation parameter
        alpha = hawkes_result.parameters.get('alpha', 0)
        if alpha > 0.1:
            effectiveness['score'] += 0.1
            effectiveness['reasons'].append("Strong self-excitation detected")
        elif alpha < 0.05:
            effectiveness['reasons'].append("Weak self-excitation - events may be independent")

        return effectiveness

    def _assess_tests_effectiveness(self, adf_result: Dict, lb_result: Dict, jb_result: Dict, regime_info: Dict) -> Dict[str, Any]:
        """Assess classical statistical tests effectiveness"""
        regime_name = regime_info.get('regime_name', '')

        effectiveness = {
            'score': 0.5,
            'reasons': [],
            'reliability': 'Medium'
        }

        # Large sample regimes
        if 'HighLiq_HighVol' in regime_name:
            effectiveness['score'] = 0.8
            effectiveness['reasons'].append("Large sample: classical tests are reliable")
            effectiveness['reliability'] = 'High'
        elif 'Low' in regime_name:  # Low liquidity or volume
            effectiveness['score'] = 0.3
            effectiveness['reasons'].append("Small sample: classical tests may be unreliable")
            effectiveness['reliability'] = 'Low'

        # Check test validity
        if not jb_result.get('is_normal', False) and 'High' in regime_name:
            effectiveness['score'] -= 0.1
            effectiveness['reasons'].append("Non-normal data affects test validity")

        if adf_result.get('is_stationary', False):
            effectiveness['score'] += 0.1
            effectiveness['reasons'].append("Stationary data: suitable for classical methods")
        else:
            effectiveness['reasons'].append("Non-stationary data may violate test assumptions")

        return effectiveness

    def _assess_overall_effectiveness(self, results: Dict, regime_info: Dict) -> Dict[str, Any]:
        """Assess overall frequentist approach effectiveness"""
        regime_name = regime_info.get('regime_name', '')
        recommended = regime_info.get('recommended_approach', '')

        # Extract individual scores
        scores = []
        if 'garch' in results['methods'] and 'effectiveness' in results['methods']['garch']:
            scores.append(results['methods']['garch']['effectiveness']['score'])
        if 'hawkes' in results['methods'] and 'effectiveness' in results['methods']['hawkes']:
            scores.append(results['methods']['hawkes']['effectiveness']['score'])
        if 'statistical_tests' in results['methods'] and 'effectiveness' in results['methods']['statistical_tests']:
            scores.append(results['methods']['statistical_tests']['effectiveness']['score'])

        overall_score = np.mean(scores) if scores else 0.5

        # Adjust based on theoretical expectation
        if 'Frequentist' in recommended:
            expected_score = 0.8
        elif 'Bayesian' in recommended:
            expected_score = 0.3
        else:  # Mixed
            expected_score = 0.6

        assessment = {
            'effectiveness_score': overall_score,
            'expected_score': expected_score,
            'matches_theory': abs(overall_score - expected_score) < 0.3,
            'summary': self._generate_effectiveness_summary(overall_score, expected_score, regime_name)
        }

        return assessment

    def _generate_effectiveness_summary(self, actual: float, expected: float, regime_name: str) -> str:
        """Generate human-readable effectiveness summary"""
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

        return f"Frequentist methods are {performance} in {regime_name} regime, performing {alignment}"