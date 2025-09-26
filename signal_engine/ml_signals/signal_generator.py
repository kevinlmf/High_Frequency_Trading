"""
Machine Learning Signal Generation Module

Implements machine learning-based signal generation using various ML models
including linear regression, ridge regression, and random forest.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class MLSignalGenerator:
    """
    Machine Learning Signal Generator

    A comprehensive signal generator that uses multiple machine learning models
    to predict price movements and generate trading signals.
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}

    def prepare_features_and_target(
        self,
        features_df: pd.DataFrame,
        price_data: pd.DataFrame,
        target_horizon: int = 1
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variable for training

        Args:
            features_df: Technical indicators features DataFrame
            price_data: Price data DataFrame
            target_horizon: Prediction time horizon (periods ahead)

        Returns:
            Tuple of cleaned features DataFrame and target Series
        """
        # Create target variable - future returns
        if 'close' in price_data.columns:
            close_prices = price_data['close']
        else:
            close_prices = price_data['close_price']

        # Calculate future returns
        target = close_prices.shift(-target_horizon) / close_prices - 1

        # Align indices
        common_index = features_df.index.intersection(target.index)
        features_aligned = features_df.loc[common_index]
        target_aligned = target.loc[common_index]

        # Remove rows containing NaN values
        mask = ~(features_aligned.isnull().any(axis=1) | target_aligned.isnull())
        features_clean = features_aligned[mask]
        target_clean = target_aligned[mask]

        # Final check for any remaining NaN values
        features_clean = features_clean.fillna(0)
        target_clean = target_clean.fillna(0)

        logger.info(f"Prepared {len(features_clean)} samples with {len(features_clean.columns)} features")

        return features_clean, target_clean

    def train_models(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        test_size: float = 0.3,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train multiple ML models

        Args:
            features: Feature matrix DataFrame
            target: Target variable Series
            test_size: Test set ratio for train/validation split
            random_state: Random state for reproducibility

        Returns:
            Training results dictionary
        """
        logger.info("Training ML models...")

        # Split data into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )

        # Scale features for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers['standard'] = scaler

        # Configure models
        models_config = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
        }

        results = {}

        for model_name, model in models_config.items():
            logger.info(f"Training {model_name} model...")

            try:
                # Train model with scaled data for linear models, original data for tree-based models
                if model_name in ['linear', 'ridge']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                # Calculate performance metrics
                r2 = r2_score(y_test, y_pred)
                ic = np.corrcoef(y_test, y_pred)[0, 1] if len(np.unique(y_pred)) > 1 else 0
                hit_rate = np.mean((y_pred * y_test) > 0) * 100

                # Save model and results
                self.models[model_name] = model
                self.performance_metrics[model_name] = {
                    'r2_score': r2,
                    'information_coefficient': ic,
                    'hit_rate': hit_rate,
                    'predictions': y_pred,
                    'actual': y_test.values
                }

                # Extract feature importance (for models that support it)
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = dict(
                        zip(features.columns, model.feature_importances_)
                    )
                elif hasattr(model, 'coef_'):
                    self.feature_importance[model_name] = dict(
                        zip(features.columns, np.abs(model.coef_))
                    )

                results[model_name] = {
                    'model': model,
                    'r2_score': r2,
                    'information_coefficient': ic,
                    'hit_rate': hit_rate
                }

                logger.info(f"{model_name}: R² = {r2:.4f}, IC = {ic:.4f}, Hit Rate = {hit_rate:.1f}%")

            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue

        return results

    def generate_signals(
        self,
        features: pd.DataFrame,
        model_name: str = 'ridge'
    ) -> pd.Series:
        """
Generate trading signals

        Args:
            features: Feature matrix DataFrame
            model_name: Model name to use for prediction

        Returns:
            Trading signals Series (-1, 0, 1)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

        model = self.models[model_name]

        # featuresHasNaN
        features_clean = features.fillna(0)

        # Processfeatures
        if model_name in ['linear', 'ridge'] and 'standard' in self.scalers:
            features_processed = self.scalers['standard'].transform(features_clean)
        else:
            features_processed = features_clean.values

        # Generate predictions
        predictions = model.predict(features_processed)

        # signals (-1, 0, 1)
        signals = pd.Series(index=features.index, data=0)
        signals[predictions > 0.001] = 1   # signals
        signals[predictions < -0.001] = -1  # signals

        return signals

    def get_signal_strength(
        self,
        features: pd.DataFrame,
        model_name: str = 'ridge'
    ) -> pd.Series:
        """
GetSignal strength ()

        Args:
            features: features            model_name: UsingModel name

        Returns:
            Signal strength"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]

        # featuresHasNaN
        features_clean = features.fillna(0)

        # Processfeatures
        if model_name in ['linear', 'ridge'] and 'standard' in self.scalers:
            features_processed = self.scalers['standard'].transform(features_clean)
        else:
            features_processed = features_clean.values

        # Generate        predictions = model.predict(features_processed)

        return pd.Series(index=features.index, data=predictions)

    def get_feature_importance(self, model_name: str, top_n: int = 20) -> Dict[str, float]:
        """
Getfeatures"""
        if model_name not in self.feature_importance:
            return {}

        importance = self.feature_importance[model_name]
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )

        return sorted_importance

    def get_performance_report(self) -> pd.DataFrame:
        """
GetHasmodelperformance"""
        if not self.performance_metrics:
            return pd.DataFrame()

        report_data = []
        for model_name, metrics in self.performance_metrics.items():
            report_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Information Coefficient': f"{metrics['information_coefficient']:.4f}",
                'Hit Rate': f"{metrics['hit_rate']:.1f}%",
                'R² Score': f"{metrics['r2_score']:.4f}",
                'Assessment': self._assess_signal_quality(metrics['information_coefficient'])
            })

        return pd.DataFrame(report_data)

    def _assess_signal_quality(self, ic: float) -> str:
        """
Evaluatesignals"""
        if ic > 0.15:
            return "Strong Signal"
        elif ic > 0.10:
            return "Good Signal"
        elif ic > 0.05:
            return "Moderate Signal"
        elif ic > 0.02:
            return "Weak Signal"
        else:
            return "Poor Signal"

    def save_models(self, filepath: str) -> None:
        """
SaveTrainmodel
"""
        import joblib
        save_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'performance_metrics': self.performance_metrics
        }
        joblib.dump(save_data, filepath)
        logger.info(f"Models saved to {filepath}")

    def load_models(self, filepath: str) -> None:
        """
LoadTrain models
"""
        import joblib
        save_data = joblib.load(filepath)
        self.models = save_data['models']
        self.scalers = save_data['scalers']
        self.feature_importance = save_data.get('feature_importance', {})
        self.performance_metrics = save_data.get('performance_metrics', {})
        logger.info(f"Models loaded from {filepath}")


if __name__ == "__main__":
    # Test    from ..data_sources.yahoo_finance import YahooFinanceSource
    from ..feature_engineering.technical_indicators import TechnicalIndicators

    # Getdata
    source = YahooFinanceSource()
    data = source.download_data("AAPL", period="5d", interval="1m")

    # Generatefeatures
    indicators = TechnicalIndicators(data)
    features = indicators.calculate_all_features()

    # Train models
    signal_gen = MLSignalGenerator()
    features_clean, target = signal_gen.prepare_features_and_target(features, data)
    results = signal_gen.train_models(features_clean, target)

    # results
    print("\n=== Model Performance ===")
    print(signal_gen.get_performance_report())