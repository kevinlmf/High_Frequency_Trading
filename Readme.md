# HFT Trading System

> **A comprehensive pipeline for generating, testing, and evaluating trading strategies with complete financial metrics analysis.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

---

## Overview
A complete high-frequency trading system that combines multiple strategy types with comprehensive financial performance analysis. The system processes real market data and generates trading signals using machine learning, traditional quantitative methods, deep learning, and advanced DeepLOB + Transformer approaches.

**Core Features**
- **Complete strategy comparison**: ML + Traditional + Deep Learning + DeepLOB + Transformer
- **Comprehensive financial metrics**: Returns, Volatility, Sharpe Ratio, Maximum Drawdown, Calmar Ratio, VaR
- **Real market data** processing via Yahoo Finance API
- **59 technical features**: price, volume, momentum, volatility indicators
- **Multiple strategy types**: Linear/Ridge/Random Forest, Momentum/Mean Reversion/Pairs Trading, LSTM/GRU, DeepLOB + Transformer
- **Professional financial analysis**: Risk-adjusted performance evaluation
- **Fast execution**: Complete analysis in under 2 seconds

---

## Quick Start

```bash
git clone git@github.com:kevinlmf/HFT_Signal.git
cd HFT_Signal
pip install pandas numpy yfinance scikit-learn

# Simple signal generation
python run_complete_pipeline.py --symbol AAPL --quick

# Complete strategy comparison (ML + Traditional + Deep Learning + DeepLOB)
python run_strategy_comparison.py --symbol AAPL --quick

# Download and use real data
python data/download_real_data.py --source enhanced_yahoo --symbol AAPL --period 5d --interval 1m
python run_strategy_comparison.py --symbol AAPL --period 5d --interval 1m

# View results in exports/ directory
```

## Project Structure
```text
HFT_Signal/
├── run_complete_pipeline.py     # Simple signal generation pipeline
├── run_strategy_comparison.py   # Complete strategy comparison (ML+Traditional+DL+DeepLOB)
├── data/                        # Data acquisition and storage
│   ├── download_real_data.py    # Script for downloading market data
│   └── real_data/               # Real market data files
├── signal_engine/               # Core signal processing
│   ├── data_sources/            # Data source integrations (Yahoo Finance, synthetic data)
│   ├── feature_engineering/     # Technical indicator calculation (59 features)
│   ├── ml_signals/              # Machine learning signal generation
│   └── signal_processor.py      # Unified signal processing pipeline
├── strategy_methods/            # Complete strategy implementations
│   ├── traditional/             # Classical quant strategies (Momentum, Mean Reversion, Pairs)
│   ├── llm_methods/             # Deep learning strategies (LSTM, GRU, Transformer, CNN-LSTM)
│   └── deep_learning_methods/   # Advanced deep learning (DeepLOB + Transformer)
├── evaluation/                  # Performance analysis and backtesting
│   ├── backtester.py            # Strategy backtesting engine
│   ├── performance_metrics.py   # Comprehensive financial metrics calculation
│   └── comparison_dashboard.py  # Strategy comparison framework
├── exports/                     # Results and reports
│   ├── *_signals.csv            # Generated trading signals
│   ├── *_performance.csv        # Model performance metrics
│   └── *_strategy_comparison.csv # Complete financial strategy comparison results
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

## Use Cases

This system provides comprehensive trading strategy research and development capabilities:

- **Quantitative Research (Factor Testing)**
  - Acts as a signal factory: generate and evaluate candidate features (X) against future returns (y).
  - Identify which indicators carry predictive power before integrating into a trading strategy.

- **Strategy Performance Analysis**
  - Compare ML, traditional, deep learning, and DeepLOB + Transformer approaches using standardized financial metrics.
  - Risk-adjusted performance evaluation with Sharpe ratio, maximum drawdown, and VaR analysis.

- **Deep Learning Feature Engineering**
  - Extracted features (e.g., VWAP, spread, RSI, volatility) serve as input for advanced neural architectures.
  - Pipeline validates which features are informative for DeepLOB and Transformer models.

- **Machine Learning Research**
  - Systematic approach to transform raw market data into actionable input signals.
  - Feature importance analysis and model performance comparison across different algorithms.

---

## System Architecture

The system follows a systematic approach combining multiple strategy types with comprehensive financial analysis:

### Target Definition (y)
```python
y = next_period_return = (price_t+1 - price_t) / price_t
```
- **Objective**: Predict future price movements
- **Horizon**: Next period return (configurable timeframe)
- **Nature**: Continuous regression for ML, classification for DeepLOB + Transformer

### Feature Engineering (X)
```python
X = [technical_indicators, volume_features, volatility_measures, momentum_indicators, ...]
```
- **59 Technical Features** across multiple dimensions:
  - **Price**: Returns, moving averages, momentum ratios
  - **Volume**: VWAP, volume ratios, price-volume products
  - **Microstructure**: Bid-ask spreads, order imbalances
  - **Technical**: RSI, Bollinger Band positions, volatility measures

### Strategy Implementation
```python
strategies = [ML_models, traditional_quant, deep_learning, deeplob_transformer]
```
- **Machine Learning**: Linear/Ridge/Random Forest with feature importance
- **Traditional Quant**: Momentum, Mean Reversion, Pairs Trading
- **Deep Learning**: LSTM, GRU with sequential modeling
- **DeepLOB + Transformer**: CNN feature extraction + attention mechanism for LOB modeling

### Financial Performance Evaluation
**Comprehensive Metrics Calculated**:
- **Return Metrics**: Annualized return, cumulative return, average daily return
- **Risk Metrics**: Volatility, maximum drawdown, VaR (5%)
- **Risk-Adjusted**: Sharpe ratio, Calmar ratio, Information ratio
- **Trading Performance**: Win rate, profit/loss ratio, signal accuracy

---

## Performance Results

### Machine Learning Signal Generation (AAPL, 1 day, 5-minute resolution):
```
Model             Info Coeff   Hit Rate   R² Score   Assessment
Linear Regression    0.4510      75.0%     -1.6433   Strong Signal
Ridge Regression     0.4475      75.0%     -0.2300   Strong Signal
Random Forest        0.4928      50.0%     -0.0313   Excellent Signal
```

### Complete Financial Strategy Comparison:
```
Strategy                  Type         Ann.Ret   Vol      Sharpe   MaxDD     Assessment
ML - Linear               Machine Lea    8.58%     1.9%     3.27     -0.48%    Good
ML - Ridge                Machine Lea    6.83%     1.8%     2.54     -0.48%    Good
LLM - LSTM                Deep Learni    3.10%     0.9%     1.20     -0.16%    Moderate
LLM - GRU                 Deep Learni    2.79%     0.9%     0.86     -0.16%    Weak
Traditional - Momentum    Rule-based     1.93%     1.4%    -0.05     -0.82%    Weak
Traditional - Mean Rev.   Rule-based    -2.79%     1.4%    -3.57     -1.48%    Weak
```

**Key Findings:**
- **Best Overall Performance**: ML - Linear (8.58% annual return, 3.27 Sharpe ratio)
- **Risk Control**: ML strategies show superior drawdown management
- **Deep Learning**: LSTM and GRU provide moderate risk-adjusted returns
- **Traditional Strategies**: Underperform in current market conditions
- **Execution Speed**: Complete analysis in under 2 seconds

---

## Technical Features

### Price Analysis
- **Returns**: 1, 5, 10-period returns and log returns
- **Moving Averages**: SMA and EMA across multiple timeframes
- **Price Ratios**: Close-to-high, close-to-low, HL ratios

### Volume & Microstructure
- **VWAP**: Volume Weighted Average Price
- **Money Flow Index**: Volume-based momentum indicator
- **Price Impact**: Estimated market impact measures
- **Spread Analysis**: High-low spread patterns

### Momentum & Technical
- **RSI**: Relative Strength Index (14-period)
- **MACD**: Moving Average Convergence Divergence
- **Stochastic**: %K and %D oscillators
- **Williams %R**: Momentum oscillator

### Volatility Measures
- **ATR**: Average True Range
- **Bollinger Bands**: Position within bands
- **Historical Volatility**: Multiple timeframe volatility
- **True Range**: Intraday volatility proxy

---

## Usage Examples

### Basic Operations
```bash
# Quick signal generation (fastest)
python run_complete_pipeline.py --symbol AAPL --quick

# Complete strategy comparison (comprehensive)
python run_strategy_comparison.py --symbol AAPL --quick

# Full comparison with all strategies
python run_strategy_comparison.py --symbol AAPL --include-all
```

### Customized Analysis
```bash
# Skip slow strategies for faster results
python run_strategy_comparison.py --symbol MSFT --skip-deeplob --skip-llm

# Compare only traditional vs ML strategies
python run_strategy_comparison.py --symbol GOOGL --skip-deeplob --skip-llm --period 2d

# Use real downloaded data
python data/download_real_data.py --source enhanced_yahoo --symbol AAPL --period 5d
python run_strategy_comparison.py --symbol AAPL --period 5d --interval 1m
```

### Advanced Applications
This pipeline serves as a **strategy research laboratory**:

- **Quantitative Research**: Systematic factor testing and validation
- **Algorithm Comparison**: ML vs Traditional vs Deep Learning vs DeepLOB + Transformer performance
- **Feature Discovery**: Identify which technical indicators drive returns
- **Strategy Selection**: Choose optimal approaches for different market conditions

## Future Extensions

- **Trading System Integration**: Connect with execution engines and risk management
- **Advanced Models**: XGBoost, attention mechanisms, ensemble methods
- **Real-Time Deployment**: Streaming data processing and live signal generation
- **Cross-Asset Analysis**: Multi-instrument and cross-market strategies

---

## Financial Performance Metrics

### Return Metrics
- **Annualized Return**: Compound annual growth rate of the strategy
- **Cumulative Return**: Total return over the evaluation period
- **Average Daily Return**: Mean daily performance

### Risk Metrics
- **Volatility**: Annualized standard deviation of returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **VaR (5%)**: Value at Risk at 5% confidence level

### Risk-Adjusted Performance
- **Sharpe Ratio**: Excess return per unit of volatility
- **Calmar Ratio**: Annual return divided by maximum drawdown
- **Information Ratio**: Excess return per unit of tracking error

### Trading Metrics
- **Win Rate**: Percentage of profitable trades/signals
- **Profit/Loss Ratio**: Average profit divided by average loss
- **Hit Rate**: Directional accuracy of predictions

### Model Evaluation
- **Information Coefficient (IC)**: Correlation between predicted and actual returns
- **R² Score**: Explained variance in return predictions
- **Feature Importance**: Ranking of predictive features

---

## Disclaimer

This system is designed for **educational and research purposes**. All trading involves risk, and past performance does not guarantee future results. Users should:

- Perform thorough backtesting before any live deployment
- Consider transaction costs, slippage, and market impact
- Understand that model performance can degrade over time
- Validate all signals with proper risk management

---

**License**: MIT | **Status**: Active Development | **Python**: 3.8+
