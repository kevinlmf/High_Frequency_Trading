"""
Signal Processor Module

Integrates data acquisition, feature engineering, and signal generation interfaces
for the HFT trading system.
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from .data_sources.yahoo_finance import YahooFinanceSource
from .data_sources.real_data_downloader import RealDataDownloader
from .feature_engineering.technical_indicators import TechnicalIndicators
from .ml_signals.signal_generator import MLSignalGenerator

logger = logging.getLogger(__name__)


class SignalProcessor:
    """
    Signal Processor Class

    Core functionality from the original HFT_Signal project that implements the
    complete signal generation pipeline including data loading, feature engineering,
    model training, and signal generation.
    """

    def __init__(self, data_source: str = 'yahoo'):
        """
        Initialize Signal Processor

        Args:
            data_source: Data source type ('yahoo', 'real_data', 'file', etc.)
        """
        self.data_source_type = data_source
        self.data_source = self._init_data_source(data_source)
        self.technical_indicators = None
        self.signal_generator = None
        self.raw_data = None
        self.features = None
        self.signals = None
        self.performance_metrics = {}

        logger.info(f"SignalProcessor initialized with {data_source} data source")

    def _init_data_source(self, source_type: str):
        """Initialize data source based on type.

        Args:
            source_type: Type of data source to initialize

        Returns:
            Initialized data source object
        """
        if source_type == 'yahoo':
            return YahooFinanceSource()
        elif source_type == 'real_data':
            return RealDataDownloader()
        else:
            raise ValueError(f"Unsupported data source: {source_type}")

    def load_data(
        self,
        symbol: str = None,
        filepath: str = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load market data

        Args:
            symbol: Stock symbol (for downloading data)
            filepath: File path (for loading from file)
            **kwargs: Additional parameters (real_data supports data_type, records, levels parameters)

        Returns:
            Raw market data DataFrame
        """
        start_time = time.time()

        if filepath:
            logger.info(f"Loading data from file: {filepath}")
            self.raw_data = self.data_source.load_data(filepath)
        elif symbol:
            if self.data_source_type == 'real_data':
                # Handle real data source with multiple data types
                data_type = kwargs.get('data_type', 'enhanced_yahoo')
                logger.info(f"Generating {data_type} data for symbol: {symbol}")

                if data_type == 'enhanced_yahoo':
                    period = kwargs.get('period', '5d')
                    interval = kwargs.get('interval', '1m')
                    self.raw_data = self.data_source.download_enhanced_yahoo_data(
                        symbol, period, interval
                    )
                elif data_type == 'synthetic_lob':
                    records = kwargs.get('records', 10000)
                    levels = kwargs.get('levels', 10)
                    start_price = kwargs.get('start_price', 100.0)
                    self.raw_data = self.data_source.generate_synthetic_lob_data(
                        symbol, start_price, records, levels
                    )
                elif data_type == 'crypto_lob':
                    records = kwargs.get('records', 20000)
                    start_price = kwargs.get('start_price', 45000.0)
                    self.raw_data = self.data_source.generate_crypto_lob_data(
                        symbol, start_price, records
                    )
                else:
                    raise ValueError(f"Unsupported real data type: {data_type}")
            else:
                # Handle yahoo finance source
                logger.info(f"Downloading data for symbol: {symbol}")
                period = kwargs.get('period', '5d')
                interval = kwargs.get('interval', '1m')
                self.raw_data = self.data_source.download_data(symbol, period, interval)
        else:
            raise ValueError("Either symbol or filepath must be provided")

        load_time = time.time() - start_time
        logger.info(f"Data loaded in {load_time:.2f} seconds. Shape: {self.raw_data.shape}")

        return self.raw_data

    def generate_features(self) -> pd.DataFrame:
        """
        Generate technical indicator features

        Returns:
            Technical features DataFrame
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        start_time = time.time()

        self.technical_indicators = TechnicalIndicators(self.raw_data)
        self.features = self.technical_indicators.calculate_all_features()

        feature_time = time.time() - start_time
        logger.info(f"Generated {len(self.features.columns)} features in {feature_time:.2f} seconds")

        return self.features

    def train_signal_models(self, test_size: float = 0.3) -> Dict[str, Any]:
        """
        Train signal generation models

        Args:
            test_size: Test set ratio for train/test split

        Returns:
            Training results dictionary
        """
        if self.features is None:
            raise ValueError("No features generated. Call generate_features() first.")

        start_time = time.time()

        self.signal_generator = MLSignalGenerator()

        # Prepare training data
        features_clean, target = self.signal_generator.prepare_features_and_target(
            self.features, self.raw_data
        )

        # Train models
        results = self.signal_generator.train_models(features_clean, target, test_size)

        train_time = time.time() - start_time
        logger.info(f"Models trained in {train_time:.2f} seconds")

        # Save performance metrics
        self.performance_metrics = self.signal_generator.performance_metrics

        return results

    def generate_signals(
        self,
        model_name: str = 'ridge',
        return_strength: bool = False
    ) -> pd.Series:
        """
        Generate trading signals

        Args:
            model_name: Model name to use for prediction
            return_strength: Whether to return signal strength instead of discrete signals

        Returns:
            Trading signals Series
        """
        if self.signal_generator is None:
            raise ValueError("No signal generator trained. Call train_signal_models() first.")

        if return_strength:
            self.signals = self.signal_generator.get_signal_strength(self.features, model_name)
        else:
            self.signals = self.signal_generator.generate_signals(self.features, model_name)

        logger.info(f"Generated signals using {model_name} model")
        return self.signals

    def run_full_pipeline(
        self,
        symbol: str = None,
        filepath: str = None,
        model_name: str = 'ridge',
        save_results: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run complete signal generation pipeline

        Args:
            symbol: Stock symbol
            filepath: Data file path
            model_name: Model name to use
            save_results: Whether to save results to file
            **kwargs: Additional parameters

        Returns:
            Pipeline execution results
        """
        pipeline_start = time.time()

        logger.info("Starting full HFT signal pipeline...")

        try:
            # 1. Load data
            self.load_data(symbol=symbol, filepath=filepath, **kwargs)

            # 2. Generate features
            self.generate_features()

            # 3. Train models
            training_results = self.train_signal_models()

            # 4. Generate signals
            signals = self.generate_signals(model_name=model_name)
            signal_strength = self.generate_signals(model_name=model_name, return_strength=True)

            # 5. Compile results
            results = {
                'raw_data': self.raw_data,
                'features': self.features,
                'signals': signals,
                'signal_strength': signal_strength,
                'training_results': training_results,
                'performance_metrics': self.performance_metrics,
                'feature_importance': self.signal_generator.get_feature_importance(model_name),
                'performance_report': self.signal_generator.get_performance_report()
            }

            # 6. Save results
            if save_results:
                self._save_results(results, symbol or 'data')

            total_time = time.time() - pipeline_start
            logger.info(f"Full pipeline completed in {total_time:.2f} seconds")

            results['execution_time'] = total_time

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

    def _save_results(self, results: Dict[str, Any], symbol: str) -> None:
        """
        Save results to files

        Args:
            results: Results dictionary to save
            symbol: Symbol name for file naming
        """
        try:
            # Create output directory
            output_dir = Path("exports")
            output_dir.mkdir(exist_ok=True)

            # Save signals
            signals_file = output_dir / f"{symbol}_signals.csv"
            if 'signals' in results and results['signals'] is not None:
                results['signals'].to_csv(signals_file)
                logger.info(f"Signals saved to {signals_file}")

            # Save performance report
            report_file = output_dir / f"{symbol}_performance_report.csv"
            if 'performance_report' in results and not results['performance_report'].empty:
                results['performance_report'].to_csv(report_file, index=False)
                logger.info(f"Performance report saved to {report_file}")

            # Save feature importance
            importance_file = output_dir / f"{symbol}_feature_importance.csv"
            if 'feature_importance' in results and results['feature_importance']:
                importance_df = pd.DataFrame(
                    list(results['feature_importance'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                importance_df.to_csv(importance_file, index=False)
                logger.info(f"Feature importance saved to {importance_file}")

        except Exception as e:
            logger.warning(f"Failed to save results: {str(e)}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get processing summary

        Returns:
            Summary dictionary with processing statistics
        """
        return {
            'data_shape': self.raw_data.shape if self.raw_data is not None else None,
            'features_count': len(self.features.columns) if self.features is not None else 0,
            'signals_generated': len(self.signals) if self.signals is not None else 0,
            'models_trained': list(self.performance_metrics.keys()) if self.performance_metrics else [],
            'best_model': self._get_best_model()
        }

    def _get_best_model(self) -> Optional[str]:
        """
        Get best performing model

        Returns:
            Name of the best model based on information coefficient
        """
        if not self.performance_metrics:
            return None

        best_model = max(
            self.performance_metrics.keys(),
            key=lambda k: self.performance_metrics[k]['information_coefficient']
        )
        return best_model


# Create__init__.pyfile
def create_init_files():
    """
    Create __init__.py files for proper package structure
    """
    init_files = [
        "signal_engine/data_sources/__init__.py",
        "signal_engine/feature_engineering/__init__.py",
        "signal_engine/ml_signals/__init__.py"
    ]

    for init_file in init_files:
        Path(init_file).touch()


if __name__ == "__main__":
    # Create __init__ files
    create_init_files()

    # Test complete pipeline
    processor = SignalProcessor()

    # Run pipeline
    results = processor.run_full_pipeline(
        symbol="AAPL",
        period="5d",
        interval="1m",
        model_name="ridge"
    )

    print("\n=== Pipeline Summary ===")
    print(f"Execution time: {results['execution_time']:.2f} seconds")
    print(f"Data shape: {results['raw_data'].shape}")
    print(f"Features generated: {len(results['features'].columns)}")

    print("\n=== Performance Report ===")
    print(results['performance_report'].to_string(index=False))