"""
Backtesting Engine for HFT Trading System
system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import warnings
from .performance_metrics import PerformanceMetrics

@dataclass
class Trade:
    """

"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    order_type: str  # 'market', 'limit', etc.
    execution_price: float  # 
    slippage: float  # 
    commission: float  # 

@dataclass
class Position:
    """

"""
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float

@dataclass
class PortfolioSnapshot:
    """

"""
    timestamp: datetime
    cash: float
    positions: Dict[str, Position]
    total_value: float
    daily_pnl: float
    total_pnl: float

class DataProvider(ABC):
    """
dataProvidesinterface
"""

    @abstractmethod
    def get_price_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
Getdata
"""
        pass

    @abstractmethod
    def get_market_data(self, symbols: List[str], timestamp: datetime) -> Dict[str, Dict[str, float]]:
        """
Getdata
"""
        pass

class SimpleDataProvider(DataProvider):
    """
dataProvidesImplementation
"""

    def __init__(self, price_data: Dict[str, pd.DataFrame]):
        """
        Args:
            price_data: {symbol: DataFrame with OHLCV data}
        """
        self.price_data = price_data

    def get_price_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        if symbol not in self.price_data:
            return pd.DataFrame()

        data = self.price_data[symbol]
        return data[(data.index >= start_date) & (data.index <= end_date)]

    def get_market_data(self, symbols: List[str], timestamp: datetime) -> Dict[str, Dict[str, float]]:
        market_data = {}
        for symbol in symbols:
            if symbol in self.price_data:
                data = self.price_data[symbol]
                # Gettimestampdata
                closest_idx = data.index.get_indexer([timestamp], method='nearest')[0]
                if closest_idx >= 0 and closest_idx < len(data):
                    row = data.iloc[closest_idx]
                    market_data[symbol] = {
                        'open': row['open'] if 'open' in row else row['price'],
                        'high': row['high'] if 'high' in row else row['price'],
                        'low': row['low'] if 'low' in row else row['price'],
                        'close': row['close'] if 'close' in row else row['price'],
                        'volume': row['volume'] if 'volume' in row else 0,
                        'price': row['close'] if 'close' in row else row['price']
                    }
        return market_data

class TradingStrategy(ABC):
    """strategyinterface"""

    @abstractmethod
    def generate_signals(self, market_data: Dict[str, Dict[str, float]],
                        portfolio: PortfolioSnapshot) -> Dict[str, float]:
        """
        Generate trading signals

        Args:
            market_data: data
            portfolio: When

        Returns:
            {symbol: target_weight}
        """
        pass

class ExecutionEngine:
    """
    Execution Engine for processing trades
    """

    def __init__(self, commission_rate: float = 0.001, slippage_model: Optional[Callable] = None):
        """
        Args:
            commission_rate: Commission rate
            slippage_model: Slippage model function
        """
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model or self._default_slippage_model

    def _default_slippage_model(self, symbol: str, quantity: float, market_data: Dict[str, float]) -> float:
        """
        Default slippage model
        """
        # model
        base_slippage = 0.0005  # 0.05%
        volume_impact = abs(quantity) * 1e-6  # 
        return base_slippage + volume_impact

    def execute_trade(self, symbol: str, side: str, quantity: float,
                     market_data: Dict[str, float], order_type: str = 'market') -> Trade:
        """
        Execute a trade
        """
        if quantity == 0:
            return None

        current_price = market_data.get('price', market_data.get('close', 0))
        slippage_rate = self.slippage_model(symbol, quantity, market_data)

        # Calculate
        if side == 'buy':
            execution_price = current_price * (1 + slippage_rate)
        else:
            execution_price = current_price * (1 - slippage_rate)

        slippage = abs(execution_price - current_price) * abs(quantity)
        commission = abs(quantity * execution_price * self.commission_rate)

        return Trade(
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            quantity=abs(quantity),
            price=current_price,
            order_type=order_type,
            execution_price=execution_price,
            slippage=slippage,
            commission=commission
        )

class Backtester:
    """
    Backtesting engine
    """

    def __init__(self,
                 data_provider: DataProvider,
                 execution_engine: ExecutionEngine,
                 initial_capital: float = 1000000.0,
                 benchmark_symbol: Optional[str] = None):
        """
        Initialize backtester

        Args:
            data_provider: Data provider
            execution_engine: Execution engine
            initial_capital: Initial capital
            benchmark_symbol: Benchmark symbol
        """
        self.data_provider = data_provider
        self.execution_engine = execution_engine
        self.initial_capital = initial_capital
        self.benchmark_symbol = benchmark_symbol

        # 
        self.cash = initial_capital
        self.positions = {}  # {symbol: Position}
        self.trades = []  # List[Trade]
        self.portfolio_history = []  # List[PortfolioSnapshot]
        self.performance_history = pd.DataFrame()

    def run_backtest(self,
                    strategy: TradingStrategy,
                    symbols: List[str],
                    start_date: datetime,
                    end_date: datetime,
                    rebalance_frequency: str = 'D') -> Dict[str, Any]:
        """
        Run backtest

        Args:
            strategy: Trading strategy
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            rebalance_frequency: Rebalance frequency ('D', 'H', 'T')

        Returns:
            Backtest results
        """
        print(f"Starting: {start_date}  {end_date}")

        # GenerateTime
        if rebalance_frequency == 'D':
            time_range = pd.date_range(start_date, end_date, freq='D')
        elif rebalance_frequency == 'H':
            time_range = pd.date_range(start_date, end_date, freq='H')
        else:
            time_range = pd.date_range(start_date, end_date, freq='T')

        # 
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []

        # Time
        for timestamp in time_range:
            self._process_timestamp(strategy, symbols, timestamp)

        # Calculatemetrics
        results = self._calculate_results(symbols)

        print(f"Completed，Process {len(time_range)} Time")
        print(f": {len(self.trades)}")

        return results

    def _process_timestamp(self, strategy: TradingStrategy, symbols: List[str], timestamp: datetime):
        """
        Process timestamp
        """
        # Getdata
        market_data = self.data_provider.get_market_data(symbols, timestamp)

        if not market_data:
            return

        # Update
        self._update_positions(market_data, timestamp)

        # CreateWhen
        portfolio_snapshot = self._create_portfolio_snapshot(timestamp)

        # Generate trading signals
        try:
            signals = strategy.generate_signals(market_data, portfolio_snapshot)
        except Exception as e:
            warnings.warn(f"strategySignal GenerationFailed at {timestamp}: {e}")
            signals = {}

        # 
        if signals:
            self._execute_rebalance(signals, market_data, timestamp)

        # 
        self.portfolio_history.append(self._create_portfolio_snapshot(timestamp))

    def _update_positions(self, market_data: Dict[str, Dict[str, float]], timestamp: datetime):
        """
Update
"""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol].get('price', market_data[symbol].get('close', 0))
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity

    def _execute_rebalance(self, signals: Dict[str, float],
                          market_data: Dict[str, Dict[str, float]], timestamp: datetime):
        """

"""
        total_value = self.cash + sum(pos.market_value for pos in self.positions.values())

        for symbol, target_weight in signals.items():
            if symbol not in market_data:
                continue

            current_price = market_data[symbol].get('price', market_data[symbol].get('close', 0))
            if current_price <= 0:
                continue

            target_value = total_value * target_weight
            current_value = self.positions.get(symbol, Position(symbol, 0, 0, 0, 0, 0)).market_value

            trade_value = target_value - current_value
            trade_quantity = trade_value / current_price

            if abs(trade_quantity) < 1e-6:  # 
                continue

            # 
            side = 'buy' if trade_quantity > 0 else 'sell'
            trade = self.execution_engine.execute_trade(
                symbol, side, abs(trade_quantity), market_data[symbol]
            )

            if trade:
                trade.timestamp = timestamp
                self.trades.append(trade)
                self._update_position(trade)

    def _update_position(self, trade: Trade):
        """
Update
"""
        symbol = trade.symbol

        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol, 0, 0, 0, 0, 0)

        position = self.positions[symbol]

        if trade.side == 'buy':
            # 
            new_quantity = position.quantity + trade.quantity
            if new_quantity > 0:
                position.avg_price = ((position.quantity * position.avg_price +
                                     trade.quantity * trade.execution_price) / new_quantity)
            position.quantity = new_quantity
            self.cash -= trade.quantity * trade.execution_price + trade.commission
        else:
            # 
            if position.quantity >= trade.quantity:
                realized_pnl = (trade.execution_price - position.avg_price) * trade.quantity
                position.realized_pnl += realized_pnl
                position.quantity -= trade.quantity
                self.cash += trade.quantity * trade.execution_price - trade.commission
            else:
                # 
                position.quantity -= trade.quantity
                self.cash += trade.quantity * trade.execution_price - trade.commission

        # 
        if abs(position.quantity) < 1e-6:
            del self.positions[symbol]

    def _create_portfolio_snapshot(self, timestamp: datetime) -> PortfolioSnapshot:
        """
Create
"""
        total_value = self.cash + sum(pos.market_value for pos in self.positions.values())

        # CalculateWhen
        if self.portfolio_history:
            daily_pnl = total_value - self.portfolio_history[-1].total_value
        else:
            daily_pnl = total_value - self.initial_capital

        total_pnl = total_value - self.initial_capital

        return PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            positions=self.positions.copy(),
            total_value=total_value,
            daily_pnl=daily_pnl,
            total_pnl=total_pnl
        )

    def _calculate_results(self, symbols: List[str]) -> Dict[str, Any]:
        """
CalculateBacktest results
"""
        if not self.portfolio_history:
            return {}

        # 
        portfolio_values = [snapshot.total_value for snapshot in self.portfolio_history]
        timestamps = [snapshot.timestamp for snapshot in self.portfolio_history]

        portfolio_series = pd.Series(portfolio_values, index=timestamps)
        returns = portfolio_series.pct_change().dropna()

        # Get
        benchmark_returns = None
        if self.benchmark_symbol:
            benchmark_data = self.data_provider.get_price_data(
                self.benchmark_symbol,
                timestamps[0],
                timestamps[-1]
            )
            if not benchmark_data.empty:
                benchmark_price_col = 'close' if 'close' in benchmark_data.columns else 'price'
                benchmark_returns = benchmark_data[benchmark_price_col].pct_change().dropna()

        # Calculatemetrics
        perf_metrics = PerformanceMetrics(returns, benchmark_returns)
        metrics = perf_metrics.calculate_all_metrics()

        # 
        trade_stats = self._calculate_trade_statistics()

        # data
        portfolio_df = pd.DataFrame([
            {
                'timestamp': snapshot.timestamp,
                'total_value': snapshot.total_value,
                'cash': snapshot.cash,
                'daily_pnl': snapshot.daily_pnl,
                'total_pnl': snapshot.total_pnl,
                'return': (snapshot.total_value / self.initial_capital - 1) * 100
            }
            for snapshot in self.portfolio_history
        ])
        portfolio_df.set_index('timestamp', inplace=True)

        return {
            'performance_metrics': metrics,
            'trade_statistics': trade_stats,
            'portfolio_history': portfolio_df,
            'trades': pd.DataFrame([
                {
                    'timestamp': trade.timestamp,
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'execution_price': trade.execution_price,
                    'slippage': trade.slippage,
                    'commission': trade.commission
                }
                for trade in self.trades
            ]),
            'final_positions': self.positions,
            'returns_series': returns,
            'benchmark_returns': benchmark_returns,
            'performance_report': perf_metrics.generate_performance_report()
        }

    def _calculate_trade_statistics(self) -> Dict[str, Any]:
        """
Calculate
"""
        if not self.trades:
            return {}

        trades_df = pd.DataFrame([
            {
                'symbol': trade.symbol,
                'side': trade.side,
                'quantity': trade.quantity,
                'slippage': trade.slippage,
                'commission': trade.commission,
                'value': trade.quantity * trade.execution_price
            }
            for trade in self.trades
        ])

        return {
            'total_trades': len(self.trades),
            'total_volume': trades_df['value'].sum(),
            'total_commission': trades_df['commission'].sum(),
            'total_slippage': trades_df['slippage'].sum(),
            'avg_trade_size': trades_df['value'].mean(),
            'avg_commission_per_trade': trades_df['commission'].mean(),
            'avg_slippage_per_trade': trades_df['slippage'].mean(),
            'trades_by_symbol': trades_df.groupby('symbol').size().to_dict(),
            'volume_by_symbol': trades_df.groupby('symbol')['value'].sum().to_dict()
        }

# ExamplestrategyImplementation
class SimpleMovingAverageStrategy(TradingStrategy):
    """
strategy
"""

    def __init__(self, short_window: int = 10, long_window: int = 30):
        self.short_window = short_window
        self.long_window = long_window
        self.price_history = {}

    def generate_signals(self, market_data: Dict[str, Dict[str, float]],
                        portfolio: PortfolioSnapshot) -> Dict[str, float]:
        signals = {}

        for symbol, data in market_data.items():
            price = data.get('price', data.get('close', 0))

            if symbol not in self.price_history:
                self.price_history[symbol] = []

            self.price_history[symbol].append(price)

            # data
            if len(self.price_history[symbol]) > self.long_window:
                self.price_history[symbol] = self.price_history[symbol][-self.long_window:]

            # Calculate
            if len(self.price_history[symbol]) >= self.long_window:
                short_ma = np.mean(self.price_history[symbol][-self.short_window:])
                long_ma = np.mean(self.price_history[symbol][-self.long_window:])

                # Generatesignals
                if short_ma > long_ma:
                    signals[symbol] = 0.5  # 50%
                elif short_ma < long_ma:
                    signals[symbol] = 0.0  # 

        return signals

# ExampleUsing
if __name__ == "__main__":
    # GenerateExampledata
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')

    # data
    price_data = {}
    for symbol in ['AAPL', 'GOOGL']:
        prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, 252))
        volumes = np.random.uniform(1000, 10000, 252)

        price_data[symbol] = pd.DataFrame({
            'open': prices * np.random.uniform(0.99, 1.01, 252),
            'high': prices * np.random.uniform(1.00, 1.05, 252),
            'low': prices * np.random.uniform(0.95, 1.00, 252),
            'close': prices,
            'volume': volumes,
            'price': prices
        }, index=dates)

    # CreatedataProvides
    data_provider = SimpleDataProvider(price_data)

    # Create
    execution_engine = ExecutionEngine(commission_rate=0.001)

    # Create
    backtester = Backtester(
        data_provider=data_provider,
        execution_engine=execution_engine,
        initial_capital=1000000,
        benchmark_symbol='AAPL'
    )

    # Createstrategy
    strategy = SimpleMovingAverageStrategy(short_window=10, long_window=30)

    # Run
    results = backtester.run_backtest(
        strategy=strategy,
        symbols=['AAPL', 'GOOGL'],
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        rebalance_frequency='D'
    )

    # Outputresults
    print("\n=== Backtest results ===")
    print(results['performance_report'])

    print("\n===  ===")
    trade_stats = results['trade_statistics']
    for key, value in trade_stats.items():
        print(f"{key}: {value}")