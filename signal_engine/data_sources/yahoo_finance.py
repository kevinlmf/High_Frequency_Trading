"""
Yahoo Finance Data source
Based onOriginal HFT_Signal project[Translated]Data acquisition[Translated]
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class YahooFinanceSource:
    """
Yahoo FinanceData acquisitionclass
"""

    def __init__(self):
        self.cache = {}

    def download_data(
        self,
        symbol: str,
        period: str = "5d",
        interval: str = "1m",
        **kwargs
    ) -> pd.DataFrame:
        """
[Translated]data

        Args:
            symbol: Stock symbol ([Translated] 'AAPL')
            period: Time[Translated] ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Time[Translated] ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')

        Returns:
            Contains OHLCV data[Translated]DataFrame
"""
        try:
            logger.info(f"Downloading {symbol} data for {period} with {interval} interval")

            # Createtickerobject
            ticker = yf.Ticker(symbol)

            # [Translated]data
            data = ticker.history(period=period, interval=interval, **kwargs)

            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            # [Translated]
            data.columns = [col.lower() for col in data.columns]

            # [Translated]timestamp[Translated]
            data.reset_index(inplace=True)
            data['timestamp'] = data['datetime'] if 'datetime' in data.columns else data.index

            # [Translated]
            column_order = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            data = data[[col for col in column_order if col in data.columns]]

            logger.info(f"Successfully downloaded {len(data)} records for {symbol}")

            return data

        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {str(e)}")
            raise

    def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """
Get[Translated]
"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'price': info.get('currentPrice', info.get('regularMarketPrice')),
                'change': info.get('regularMarketChange'),
                'change_percent': info.get('regularMarketChangePercent'),
                'volume': info.get('regularMarketVolume'),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting real-time quote for {symbol}: {str(e)}")
            return {}

    def save_data(self, data: pd.DataFrame, filepath: str) -> None:
        """Save data to file"""
        try:
            data.to_csv(filepath, index=False)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from file"""
        try:
            data = pd.read_csv(filepath)
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            logger.info(f"Data loaded from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise


if __name__ == "__main__":
    # Test[Translated]
    source = YahooFinanceSource()
    data = source.download_data("AAPL", period="1d", interval="1m")
    print(f"Downloaded {len(data)} records")
    print(data.head())