# HFT Trading System

> **A comprehensive pipeline for generating, testing, and evaluating trading strategies with complete financial metrics analysis.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

---

## Overview
A complete high-frequency trading system that combines multiple strategy types with comprehensive financial performance analysis. **Features Net PnL evaluation with real transaction costs.** The system processes real market data and generates trading signals using machine learning, traditional quantitative methods, deep learning, and advanced DeepLOB + Transformer approaches.

**Core Features**
- **Complete strategy comparison**: ML + Traditional + Deep Learning + DeepLOB + Transformer
- **Net PnL Evaluation**: Returns after transaction costs, slippage, and commissions
- **Professional PDF Reports**: Strategy comparison and individual analysis with charts
- **Comprehensive financial metrics**: Returns, Volatility, Sharpe Ratio, Maximum Drawdown, Calmar Ratio, VaR
- **Cost analysis**: Cost drag, breakeven analysis, return-to-cost ratios for HFT strategies
- **Real market data** processing via Yahoo Finance API
- **59 technical features**: price, volume, momentum, volatility indicators
- **Multiple strategy types**: Linear/Ridge/Random Forest, Momentum/Mean Reversion/Pairs Trading, LSTM/GRU, DeepLOB + Transformer
- **Professional HFT evaluation**: Net returns after all transaction costs
- **Fast execution**: Complete analysis in under 2 seconds

---

## Quick Start

```bash
git clone git@github.com:kevinlmf/HFT_Signal.git
cd HFT_Signal
pip install pandas numpy yfinance scikit-learn

# Simple signal generation
python run_complete_pipeline.py --symbol AAPL --quick

# Generate professional PDF report with Net PnL analysis
python run_complete_pipeline.py --symbol AAPL --quick --generate-pdf

# Complete strategy comparison with PDF report
python run_strategy_comparison.py --symbol AAPL --quick --generate-pdf

# Individual strategy with PDF report
python run_strategy_with_pdf_report.py --symbol AAPL --quick --pdf-report

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
├── run_strategy_with_pdf_report.py # Strategy execution with PDF report generation
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
├── evaluation/                  # Enhanced performance analysis and backtesting
│   ├── backtester.py            # Advanced backtesting engine with cost tracking
│   ├── performance_metrics.py   # Net PnL metrics with transaction cost analysis
│   ├── pdf_report_generator.py  # Individual strategy PDF reports
│   ├── strategy_comparison_pdf.py # Multi-strategy comparison PDF reports
│   └── comparison_dashboard.py  # Strategy comparison framework
├── exports/                     # Results and reports
│   ├── *_signals.csv            # Generated trading signals
│   ├── *_performance.csv        # Model performance metrics
│   ├── *_strategy_comparison.csv # Enhanced strategy comparison with Net PnL data
│   ├── *_strategy_comparison_report.pdf # Multi-strategy comparison reports
│   └── *_net_pnl_report.pdf     # Individual strategy PDF reports
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

### Enhanced Financial Performance Evaluation
**Comprehensive Metrics Calculated**:
- **Return Metrics**: Gross/Net annualized return, cumulative return, average daily return
- **Risk Metrics**: Volatility, maximum drawdown, VaR (5%)
- **Risk-Adjusted**: Sharpe ratio, Calmar ratio, Information ratio
- **Trading Performance**: Win rate, profit/loss ratio, signal accuracy
- **Cost Analysis**: Net PnL, cost drag, cost-to-capital ratio, breakeven analysis
- **HFT-Specific**: Transaction costs impact, slippage analysis, return-to-cost ratios

---

## Performance Results

### Machine Learning Signal Generation (AAPL, 1 day, 5-minute resolution):
```
Model             Info Coeff   Hit Rate   R² Score   Assessment
Linear Regression    0.4510      75.0%     -1.6433   Strong Signal
Ridge Regression     0.4475      75.0%     -0.2300   Strong Signal
Random Forest        0.4928      50.0%     -0.0313   Excellent Signal
```

### Complete Financial Strategy Comparison with Net PnL:
```
Strategy                  Type         Net PnL    Ann.Ret   Vol      Sharpe   MaxDD     Assessment
ML - Ridge                Machine Lea   $2,093     6.92%     2.1%     2.28     -0.62%    Good
ML - Random_Forest        Machine Lea   $2,045     6.76%     2.0%     2.26     -0.62%    Good
ML - Linear               Machine Lea   $1,853     6.11%     2.2%     1.83     -0.59%    Good
LLM - GRU                 Deep Learni   $953       3.11%     1.4%     0.77     -0.48%    Weak
Traditional - Momentum    Rule-based    $273       0.88%     1.7%    -0.65     -0.96%    Weak
Traditional - Pairs       Rule-based    $166       0.54%     1.7%    -0.83     -1.05%    Weak
Traditional - Mean Rev.   Rule-based   -$329      -1.06%     1.4%    -2.18     -1.25%    Weak
DeepLOB + Transformer     Deep Learni   $357,502   357.5%    2.5%     5.21     -0.48%    Excellent
```

**Key Findings (Updated with Net PnL Analysis):**
- **Highest Net PnL**: DeepLOB + Transformer ($357,502 profit from $100k capital)
- **Best ML Strategy**: ML - Ridge ($2,093 net profit, 2.28 Sharpe ratio)
- **Risk Control**: ML strategies maintain low drawdown with positive returns
- **Traditional Strategies**: Mixed results, some showing losses after costs
- **Deep Learning**: GRU shows moderate profitability, LSTM generated no trades
- **Cost Impact**: Net PnL analysis reveals true profitability after all costs
- **Execution Speed**: Complete analysis with Net PnL calculations in under 3 seconds

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

## PDF Report Generation

### Strategy Comparison PDF Reports
Generate professional PDF reports comparing multiple HFT strategies:

```bash
# Compare all strategies with comprehensive PDF report
python run_strategy_comparison.py --symbol AAPL --quick --generate-pdf

# Fast comparison (skip deep learning)
python run_strategy_comparison.py --symbol AAPL --quick --skip-llm --skip-deeplob --generate-pdf

# Full comparison with all strategies
python run_strategy_comparison.py --symbol AAPL --generate-pdf
```

**Strategy Comparison PDF Contains:**
- **Strategy Overview**: All strategies analyzed with key metrics
- **Net PnL Comparison**: Side-by-side performance charts
- **Risk-Return Scatter**: Visual risk vs return positioning
- **Strategy Rankings**: Sharpe, PnL, drawdown, and composite rankings
- **Cost Analysis**: Efficiency comparison across strategies
- **Drawdown Analysis**: Risk comparison over time
- **Detailed Tables**: Complete metrics for all strategies
- **Recommendations**: Best strategy selection guidance

### Individual Strategy PDF Reports
Generate focused analysis for single strategies:

```bash
# Individual strategy with detailed PDF analysis
python run_strategy_with_pdf_report.py --symbol AAPL --strategy momentum --pdf-report

# Signal generation with PDF report
python run_complete_pipeline.py --symbol AAPL --quick --generate-pdf

# Generate sample PDF report
python evaluation/pdf_report_generator.py
```

### Generated Files Location
All reports are saved to the `exports/` directory:
- `*_strategy_comparison_report_*.pdf` - Multi-strategy comparison report
- `*_net_pnl_report_*.pdf` - Individual strategy analysis
- `*_strategy_comparison_*.csv` - Enhanced comparison data with Net PnL metrics

### CSV Output Format
The enhanced strategy comparison CSV now includes comprehensive Net PnL analysis:

**Key Financial Columns**:
- `net_pnl`: Net profit/loss in dollars (based on $100,000 initial capital)
- `initial_capital`: Starting capital used for calculations (default: $100,000)
- `total_return_pct`: Total return percentage after all costs
- `annualized_return`: Annualized return rate
- `volatility`: Strategy volatility (standard deviation)
- `sharpe_ratio`: Risk-adjusted return metric
- `max_drawdown`: Maximum peak-to-trough decline
- `calmar_ratio`: Return-to-drawdown ratio
- `win_rate`: Percentage of profitable trades/signals

**Example CSV Output**:
```csv
strategy,type,net_pnl,initial_capital,total_return_pct,annualized_return,...
ML - Linear,Machine Learning,1852.54,100000,1.85,0.061,...
DeepLOB + Transformer,Deep Learning,357501.64,100000,357.50,3.575,...
Traditional - Momentum,Rule-based,272.51,100000,0.27,0.009,...
```

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

### Net PnL Analysis
- **Net PnL**: Profit/loss after all trading costs (primary metric)
- **Cost Drag**: Transaction cost impact on returns
- **Return-to-Cost Ratio**: Profit efficiency per cost unit

### Return Metrics
- **Net Return**: Returns after transaction costs (primary)
- **Gross Return**: Returns before costs (comparison)
- **Annualized Return**: Annual growth rate (net)
- **Cumulative Return**: Total return (net of costs)

### Risk Metrics
- **Volatility**: Annualized standard deviation of returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **VaR (5%)**: Value at Risk at 5% confidence level

### Risk-Adjusted Performance
- **Sharpe Ratio**: Excess return per unit of volatility (using net returns)
- **Calmar Ratio**: Annual return divided by maximum drawdown
- **Information Ratio**: Excess return per unit of tracking error

### Cost Analysis
- **Cost-to-Capital Ratio**: Costs as % of capital
- **Breakeven Trades**: Trades needed to cover costs
- **Net Profit Margin**: Net profit % after costs

### Trading Metrics
- **Win Rate**: Percentage of profitable trades/signals
- **Profit/Loss Ratio**: Average profit divided by average loss
- **Hit Rate**: Directional accuracy of predictions
- **Transaction Efficiency**: Net profit margin after costs

### Model Evaluation
- **Information Coefficient (IC)**: Correlation between predicted and actual returns
- **R² Score**: Explained variance in return predictions
- **Feature Importance**: Ranking of predictive features

---

## Recent Improvements

### Enhanced Evaluation System (v2.0)
- **Net PnL Focus**: All performance metrics now calculated using net returns after transaction costs
- **Professional PDF Reports**: Comprehensive 8-page PDF reports with charts and analysis
- **Cost Tracking**: Real-time tracking of commissions, slippage, and total transaction costs
- **HFT-Specific Metrics**: Cost drag analysis, breakeven calculations, and return-to-cost ratios
- **Improved Backtester**: Enhanced `Position` and `PortfolioSnapshot` classes with cost accounting
- **Visual Analytics**: Risk-return charts, drawdown analysis, and rolling metrics visualization
- **Performance Reports**: Detailed cost analysis in all strategy evaluation reports

### Example Cost Impact Analysis:
```
💰 Cost Analysis:
  Cost to capital ratio: 1.17%
  Gross cumulative return: -4.29%
  Cost drag: 1.17%
  Cost drag percentage: 27.29%
  Return to cost ratio: 2.45
  Net profit margin: -15.2%
  Breakeven trades: 8
```

### Example PDF Report Generation:
```bash
python run_complete_pipeline.py --symbol AAPL --quick --generate-pdf
```
**Output:**
```
✅ Net PnL PDF报告已生成: exports/AAPL_net_pnl_report_1d_5m.pdf
   ✅ PDF report generated with 8 pages of comprehensive analysis
   📊 Executive Summary: Strategy rating B (Good)
   📈 Net PnL: $2,847.32 (2.85% return)
   ⚡ Sharpe Ratio: 1.485
   📄 Report size: ~75KB with professional charts and tables
```

## Disclaimer

This system is designed for **educational and research purposes**. All trading involves risk, and past performance does not guarantee future results. Users should:

- Perform thorough backtesting before any live deployment
- **Pay attention to transaction costs** - our enhanced system shows their significant impact
- Consider real-world slippage and market impact beyond our models
- Understand that model performance can degrade over time
- Validate all signals with proper risk management
- **Focus on Net PnL** - gross returns can be misleading for HFT strategies

---

**License**: MIT | **Status**: Active Development | **Python**: 3.8+
