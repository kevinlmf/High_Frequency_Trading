"""
Real High-Frequency Trading Data Downloader
==========================================

This module provides real HFT data downloading capabilities integrated with the existing HFT system.
Supports multiple data providers and generates realistic LOB data for research purposes.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
import os
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class RealDataDownloader:
    """Download and process real HFT data from various sources"""

    def __init__(self, data_dir: str = "data/real_data"):
        """Initialize downloader with data directory"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Compatible with existing YahooFinanceSource
        self.cache = {}

        # Data source configurations
        self.sources = {
            'enhanced_yahoo': {
                'name': 'Enhanced Yahoo Finance',
                'description': 'Yahoo Finance data enhanced with LOB-like features',
                'max_interval': '1m',
                'max_period': '60d'
            },
            'synthetic_lob': {
                'name': 'Synthetic LOB Data',
                'description': 'Realistic limit order book data for backtesting',
                'levels': 10,
                'frequency': '100ms'
            },
            'crypto_lob': {
                'name': 'Crypto LOB Simulation',
                'description': 'Cryptocurrency-style order book data',
                'levels': 20,
                'frequency': '50ms'
            },
            'high_freq_enhanced': {
                'name': 'High Frequency Enhanced Data',
                'description': 'Microsecond-level synthetic data with realistic microstructure',
                'frequency': '10ms',
                'features': ['bid_ask_spread', 'order_flow', 'market_impact']
            }
        }

    def download_enhanced_yahoo_data(self, symbol: str, period: str = "5d",
                                   interval: str = "1m", **kwargs) -> pd.DataFrame:
        """Download Yahoo Finance data enhanced with HFT features"""
        logger.info(f"Downloading enhanced {symbol} data from Yahoo Finance...")

        try:
            # Use yfinance to get base data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval, prepost=True, **kwargs)

            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            # Enhance with HFT features
            enhanced_data = self._enhance_yahoo_data(data, symbol)

            # Save to cache and file
            cache_key = f"{symbol}_{period}_{interval}"
            self.cache[cache_key] = enhanced_data

            output_path = self.data_dir / f"{symbol}_{period}_{interval}_enhanced.csv"
            enhanced_data.to_csv(output_path, index=False)

            logger.info(f"Enhanced data saved: {output_path}")
            return enhanced_data

        except Exception as e:
            logger.error(f"Failed to download enhanced Yahoo data: {e}")
            raise

    def generate_synthetic_lob_data(self, symbol: str = "SYNTHETIC",
                                  start_price: float = 100.0,
                                  num_records: int = 10000,
                                  levels: int = 10,
                                  frequency_ms: int = 100) -> pd.DataFrame:
        """Generate realistic synthetic limit order book data"""
        logger.info(f"Generating synthetic LOB data for {symbol}...")

        try:
            # Generate time series with specified frequency
            start_time = pd.Timestamp.now() - pd.Timedelta(minutes=num_records * frequency_ms / 60000)
            timestamps = pd.date_range(start_time, periods=num_records,
                                     freq=f'{frequency_ms}ms')

            data = []
            current_price = start_price

            for i, ts in enumerate(timestamps):
                # Realistic price movement with microstructure noise
                trend = np.sin(i / 1000) * 0.001  # Long-term trend
                noise = np.random.normal(0, 0.0005)  # Microstructure noise
                jump = np.random.exponential(0.001) * np.random.choice([-1, 1]) \
                       if np.random.random() < 0.01 else 0  # Occasional jumps

                price_change = (trend + noise + jump) * current_price
                current_price += price_change

                # Generate realistic bid-ask spread
                spread_bps = max(1, np.random.exponential(3) + 1)  # 1-10 bps typically
                spread = current_price * spread_bps / 10000
                mid_price = current_price

                record = {
                    'timestamp': ts,
                    'symbol': symbol,
                    'mid_price': round(mid_price, 4),
                    'spread': round(spread, 4),
                    'spread_bps': round(spread_bps, 2),
                    # Add OHLCV compatibility
                    'open': round(mid_price - spread/4 + np.random.normal(0, spread/8), 4),
                    'high': round(mid_price + spread/2 + np.random.exponential(spread/4), 4),
                    'low': round(mid_price - spread/2 - np.random.exponential(spread/4), 4),
                    'close': round(mid_price, 4),
                    'volume': int(np.random.exponential(10000) + 1000)
                }

                # Generate order book levels
                total_bid_volume = 0
                total_ask_volume = 0

                for level in range(1, levels + 1):
                    # Bid side - exponentially decreasing size
                    bid_price = mid_price - spread/2 - (level-1) * spread * 0.5
                    bid_size = max(100, np.random.exponential(2000 / level))

                    record[f'bid_price_{level}'] = round(bid_price, 4)
                    record[f'bid_size_{level}'] = int(bid_size)
                    total_bid_volume += bid_size

                    # Ask side
                    ask_price = mid_price + spread/2 + (level-1) * spread * 0.5
                    ask_size = max(100, np.random.exponential(2000 / level))

                    record[f'ask_price_{level}'] = round(ask_price, 4)
                    record[f'ask_size_{level}'] = int(ask_size)
                    total_ask_volume += ask_size

                # Add volume metrics
                record['total_bid_volume'] = total_bid_volume
                record['total_ask_volume'] = total_ask_volume
                record['volume_imbalance'] = (total_bid_volume - total_ask_volume) / \
                                           (total_bid_volume + total_ask_volume)

                # Add trade data
                if np.random.random() < 0.3:  # 30% chance of trade
                    trade_price = mid_price + np.random.normal(0, spread/4)
                    trade_size = np.random.exponential(500) + 100
                    record['trade_price'] = round(trade_price, 4)
                    record['trade_size'] = int(trade_size)
                    record['trade_side'] = 1 if trade_price > mid_price else -1
                else:
                    record['trade_price'] = np.nan
                    record['trade_size'] = np.nan
                    record['trade_side'] = np.nan

                data.append(record)

            df = pd.DataFrame(data)

            # Save to file
            output_path = self.data_dir / f"{symbol}_synthetic_lob_{num_records}.csv"
            df.to_csv(output_path, index=False)

            logger.info(f"Generated {len(df)} synthetic LOB records")
            return df

        except Exception as e:
            logger.error(f"Failed to generate synthetic LOB data: {e}")
            raise

    def generate_crypto_lob_data(self, symbol: str = "BTCUSDT",
                               start_price: float = 45000.0,
                               num_records: int = 20000) -> pd.DataFrame:
        """Generate cryptocurrency-style LOB data with deeper books"""
        logger.info(f"Generating crypto LOB data for {symbol}...")

        try:
            start_time = pd.Timestamp.now() - pd.Timedelta(minutes=num_records * 0.05 / 60)
            timestamps = pd.date_range(start_time, periods=num_records, freq='50ms')

            data = []
            current_price = start_price

            for i, ts in enumerate(timestamps):
                # Crypto has higher volatility
                volatility = 0.002  # 0.2% per update
                price_change = np.random.normal(0, volatility) * current_price
                current_price = max(1000, current_price + price_change)  # Floor at $1000

                # Tighter spreads for major crypto pairs
                spread = max(0.1, np.random.exponential(2.0))
                mid_price = current_price

                record = {
                    'timestamp': ts,
                    'symbol': symbol,
                    'mid_price': round(mid_price, 1),
                    'spread': round(spread, 1),
                    # Add OHLCV compatibility
                    'open': round(mid_price - spread/4 + np.random.normal(0, spread/8), 1),
                    'high': round(mid_price + spread/2 + np.random.exponential(spread/4), 1),
                    'low': round(mid_price - spread/2 - np.random.exponential(spread/4), 1),
                    'close': round(mid_price, 1),
                    'volume': round(np.random.exponential(100.0) + 10.0, 4)
                }

                # Generate 20 levels for crypto (deeper books)
                total_bid_volume = 0
                total_ask_volume = 0

                for level in range(1, 21):
                    # Bid side
                    bid_price = mid_price - spread/2 - (level-1) * 0.5
                    bid_size = max(0.001, np.random.exponential(1.0 / level))

                    record[f'bid_price_{level}'] = round(bid_price, 1)
                    record[f'bid_size_{level}'] = round(bid_size, 6)
                    total_bid_volume += bid_size

                    # Ask side
                    ask_price = mid_price + spread/2 + (level-1) * 0.5
                    ask_size = max(0.001, np.random.exponential(1.0 / level))

                    record[f'ask_price_{level}'] = round(ask_price, 1)
                    record[f'ask_size_{level}'] = round(ask_size, 6)
                    total_ask_volume += ask_size

                # Volume metrics
                record['total_bid_volume'] = round(total_bid_volume, 6)
                record['total_ask_volume'] = round(total_ask_volume, 6)
                record['volume_imbalance'] = round((total_bid_volume - total_ask_volume) /
                                                 (total_bid_volume + total_ask_volume), 4)

                # High frequency trade simulation
                if np.random.random() < 0.4:  # 40% chance of trade (crypto is active)
                    trade_price = mid_price + np.random.normal(0, spread/3)
                    trade_size = np.random.exponential(0.5)
                    record['trade_price'] = round(trade_price, 1)
                    record['trade_size'] = round(trade_size, 6)
                    record['trade_side'] = 1 if trade_price > mid_price else -1

                data.append(record)

            df = pd.DataFrame(data)

            # Save to file
            output_path = self.data_dir / f"{symbol}_crypto_lob_{num_records}.csv"
            df.to_csv(output_path, index=False)

            logger.info(f"Generated {len(df)} crypto LOB records")
            return df

        except Exception as e:
            logger.error(f"Failed to generate crypto LOB data: {e}")
            raise

    def _enhance_yahoo_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Enhance Yahoo Finance data with HFT-relevant features"""
        enhanced = data.copy()
        enhanced.reset_index(inplace=True)

        # Rename columns to match HFT system conventions
        column_mapping = {
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        enhanced.rename(columns=column_mapping, inplace=True)

        # Add symbol
        enhanced['symbol'] = symbol

        # Add microsecond precision to timestamps
        if 'timestamp' in enhanced.columns:
            enhanced['timestamp'] = enhanced['timestamp'].apply(
                lambda x: x.replace(microsecond=np.random.randint(0, 999999))
            )

        # Calculate HFT-relevant features
        enhanced['mid_price'] = (enhanced['high'] + enhanced['low']) / 2
        enhanced['spread'] = enhanced['high'] - enhanced['low']
        enhanced['spread_bps'] = (enhanced['spread'] / enhanced['mid_price'] * 10000).round(2)

        # Price and volume features
        enhanced['returns'] = enhanced['close'].pct_change()
        enhanced['log_returns'] = np.log(enhanced['close']).diff()
        enhanced['volatility'] = enhanced['returns'].rolling(20).std()

        # Simulate bid-ask based on OHLC
        enhanced['bid'] = enhanced['close'] - enhanced['spread'] * 0.3
        enhanced['ask'] = enhanced['close'] + enhanced['spread'] * 0.3

        # Volume features
        enhanced['volume_ma_5'] = enhanced['volume'].rolling(5).mean()
        enhanced['volume_ratio'] = enhanced['volume'] / enhanced['volume_ma_5']
        enhanced['price_volume'] = enhanced['close'] * enhanced['volume']

        # Microstructure indicators
        enhanced['price_impact'] = np.abs(enhanced['returns']) * np.log(enhanced['volume'])
        enhanced['order_flow_imbalance'] = np.random.normal(0, 0.1, len(enhanced))  # Simulated

        # Generate synthetic LOB levels (simplified)
        for level in range(1, 6):  # 5 levels
            bid_offset = level * enhanced['spread'] * 0.2
            ask_offset = level * enhanced['spread'] * 0.2

            enhanced[f'bid_price_{level}'] = (enhanced['bid'] - bid_offset).round(4)
            enhanced[f'ask_price_{level}'] = (enhanced['ask'] + ask_offset).round(4)
            enhanced[f'bid_size_{level}'] = (enhanced['volume'] * 0.1 *
                                           np.random.exponential(1/level, len(enhanced))).astype(int)
            enhanced[f'ask_size_{level}'] = (enhanced['volume'] * 0.1 *
                                           np.random.exponential(1/level, len(enhanced))).astype(int)

        return enhanced

    def get_available_sources(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available data sources"""
        return self.sources

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from file - compatible with existing system"""
        try:
            data = pd.read_csv(filepath)
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            logger.info(f"Data loaded from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def save_data(self, data: pd.DataFrame, filepath: str) -> None:
        """Save data to file - compatible with existing system"""
        try:
            data.to_csv(filepath, index=False)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

    def get_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive information about the dataset"""
        info = {
            'num_records': len(data),
            'num_columns': len(data.columns),
            'columns': list(data.columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'date_range': {
                'start': data['timestamp'].min() if 'timestamp' in data.columns else 'N/A',
                'end': data['timestamp'].max() if 'timestamp' in data.columns else 'N/A'
            }
        }

        # Add price statistics if available
        if 'mid_price' in data.columns:
            info['price_stats'] = {
                'min': data['mid_price'].min(),
                'max': data['mid_price'].max(),
                'mean': data['mid_price'].mean(),
                'std': data['mid_price'].std()
            }

        # Add volume statistics if available
        if 'volume' in data.columns:
            info['volume_stats'] = {
                'total': data['volume'].sum(),
                'mean': data['volume'].mean(),
                'max': data['volume'].max()
            }

        return info


if __name__ == "__main__":
    # Test the downloader
    downloader = RealDataDownloader()

    # Test enhanced Yahoo Finance data
    try:
        yahoo_data = downloader.download_enhanced_yahoo_data("AAPL", period="1d", interval="1m")
        print(f"Enhanced Yahoo data: {len(yahoo_data)} records")
        print(yahoo_data.head())
    except Exception as e:
        print(f"Yahoo data test failed: {e}")

    # Test synthetic LOB data
    try:
        lob_data = downloader.generate_synthetic_lob_data("TEST", num_records=1000)
        print(f"Synthetic LOB data: {len(lob_data)} records")
        print(lob_data.head())
    except Exception as e:
        print(f"Synthetic LOB test failed: {e}")