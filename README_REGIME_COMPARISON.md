# HFT Regime-Based Statistical Approach Comparison System

## Overview

This system implements your innovative idea of comparing **Frequentist vs Bayesian** statistical approaches across **9 different HFT market regimes**, based on the three core market environment factors:

- **Liquidity** (High/Medium/Low)
- **Volume** (High/Low)
- **Volatility** (High/Low)

The system empirically tests which statistical approach works better in each of the 9 regime combinations, validating your theoretical 9-grid framework.

## System Architecture

### Core Components

1. **Market Regime Classifier** (`regime_classification/`)
   - Automatically classifies market into 9 regimes
   - Uses liquidity proxies, volume intensity, and volatility patterns
   - Configurable classification thresholds

2. **Frequentist Methods** (`statistical_methods/frequentist_methods.py`)
   - GARCH models for volatility modeling
   - Hawkes processes for event clustering
   - Classical statistical tests (ADF, Ljung-Box, Jarque-Bera)
   - Maximum likelihood estimation

3. **Bayesian Methods** (`statistical_methods/bayesian_methods.py`)
   - Hierarchical Bayesian models for uncertainty quantification
   - Markov Switching models for regime changes
   - Dynamic Bayesian updating for online learning
   - Variational inference for fast computation

4. **Comparison Framework** (`regime_comparison/`)
   - Integrated analysis across all regimes
   - Theory vs empirical validation
   - Performance metrics and uncertainty quantification
   - Comprehensive reporting and visualization

## Your 9-Grid Theoretical Framework

| Regime | Liquidity | Volume | Volatility | Expected Winner | Reasoning |
|--------|-----------|--------|------------|----------------|-----------|
| 1 | High | High | High | **Bayesian** | 不确定性更关键，需要跳跃/状态切换先验 |
| 2 | High | High | Low | **Frequentist** | 大样本稳定，参数估计快速可靠 |
| 3 | High | Low | High | **Bayesian** | 成交稀疏噪声大，层次贝叶斯处理稀疏数据 |
| 4 | High | Low | Low | **Frequentist** | 冷清但盘口稳，经典方法简单够用 |
| 5 | Medium | High | High | **Mixed** | 主流币状态，Frequentist打底+Bayesian调优 |
| 6 | Medium | High | Low | **Frequentist** | 足够样本，参数估计可靠 |
| 7 | Medium | Low | High | **Bayesian** | 样本不足，先验知识稳定估计 |
| 8 | Low | High | High | **Bayesian** | 小币爆发，动态先验适应regime shift |
| 9 | Low | Low | High | **Bayesian** | 山寨币典型，稀少数据需要先验建模 |

## Quick Start

### 1. Demo Run
```bash
# Run with synthetic data to see the system in action
python demo_regime_comparison.py
```

### 2. Real Market Analysis
```bash
# Basic analysis
python run_regime_statistical_comparison.py --symbol AAPL --period 5d --interval 5m

# Quick analysis (faster)
python run_regime_statistical_comparison.py --symbol BTCUSDT --quick --export-csv

# Full comprehensive analysis
python run_regime_statistical_comparison.py --symbol TSLA --period 10d --full-analysis --generate-report
```

### 3. Custom Data
```bash
# Use your own OHLCV CSV data
python run_regime_statistical_comparison.py --file data/my_hft_data.csv --save-plots
```

## Output Examples

### Console Output
```
HFT REGIME ANALYSIS COMPLETE: AAPL
============================================================
Regimes Analyzed: 6
Theory Validation: 83.3%
Results: Freq=2, Bayes=3, Ties=1
Recommendation: Bayesian methods generally preferred (high theoretical alignment)

Key Insights:
   • Strong theoretical validation - HFT regime theory holds well
   • Bayesian methods consistently better in high volatility regimes
   • Bayesian methods superior with small sample sizes
============================================================
```

### Generated Files
- **CSV Export**: Detailed numerical results for further analysis
- **Visualization**: 2x2 subplot comparison matrix showing:
  - Method effectiveness by regime
  - Theory vs empirical performance scatter
  - Sample size effects on method performance
  - Performance breakdown by liquidity level
- **Text Report**: Comprehensive analysis summary
- **JSON Results**: Full programmatic results for integration

## Statistical Methods Details

### Frequentist Arsenal
- **GARCH(1,1)**: Volatility clustering detection
- **Hawkes Process**: Self-exciting event modeling
- **Classical Tests**: Stationarity, autocorrelation, normality
- **MLE Estimation**: Maximum likelihood parameter fitting

**Best Performance**: High liquidity + high volume regimes where large samples provide stable parameter estimates.

### Bayesian Arsenal
- **Hierarchical Bayesian**: Borrows strength across similar periods/assets
- **Markov Switching**: Handles regime changes and structural breaks
- **Dynamic Updating**: Online learning with exponential forgetting
- **Variational Inference**: Fast approximate posterior computation

**Best Performance**: Low volume, high volatility regimes where uncertainty quantification and prior knowledge are crucial.

## Key Metrics Tracked

### Effectiveness Scores (0-1 scale)
- Method appropriateness for regime characteristics
- Parameter estimation reliability
- Statistical test validity

### Performance Comparison
- Prediction accuracy (RMSE, MAE)
- Uncertainty quantification quality
- Computational efficiency
- Theory validation rate

### Regime Classification Quality
- Regime persistence and stability
- Transition patterns
- Classification confidence

## Configuration Options

### Regime Classification Parameters
```python
regime_params = {
    'lookback_period': 100,          # Rolling window for metrics
    'liquidity_thresholds': (0.33, 0.67),  # Tercile boundaries
    'volume_threshold': 0.5,         # Binary volume classification
    'volatility_threshold': 0.5      # Binary volatility classification
}
```

### Analysis Options
- `--quick`: Faster analysis with reduced complexity
- `--full-analysis`: Most comprehensive analysis
- `--export-csv`: Export detailed numerical results
- `--save-plots`: Generate visualization files
- `--generate-report`: Create comprehensive text report

## Validation Results Interpretation

### High Theory Validation (>70%)
**Your 9-grid framework is empirically validated**
- Market regimes behave as theoretically expected
- Statistical approach selection can follow your framework
- Confidence in regime-specific recommendations

### Moderate Validation (50-70%)
**Partial validation with some surprises**
- Most regimes behave as expected
- Some regimes may need sub-classification
- Consider market-specific factors

### Low Validation (<50%)
**Framework needs refinement**
- Empirical results contradict theory significantly
- Consider additional regime dimensions
- May indicate data quality or classification issues

## Extension Possibilities

### Custom Statistical Methods
```python
# Add your own methods by inheriting from base classes
class MyFrequentistMethod(FrequentistMethod):
    def fit(self, data):
        # Your implementation
        pass

class MyBayesianMethod(BayesianMethod):
    def fit(self, data, prior_params=None):
        # Your implementation
        pass
```

### Custom Regime Classification
```python
# Extend with additional market dimensions
class ExtendedRegimeClassifier(MarketRegimeClassifier):
    def calculate_custom_metric(self, data):
        # Your custom market state metric
        pass
```

## Research Applications

### Academic Research
- Empirical validation of statistical method selection
- Market microstructure regime analysis
- Uncertainty quantification in financial modeling

### Practical Trading
- Regime-adaptive statistical modeling
- Risk management approach selection
- Model validation and backtesting

### System Integration
- Real-time regime detection
- Automated method selection
- Performance monitoring and alerting

## Contributing

This system implements your innovative research idea. To extend or improve:

1. **New Statistical Methods**: Add to `statistical_methods/`
2. **Enhanced Regime Classification**: Extend `regime_classification/`
3. **Additional Visualizations**: Enhance `regime_comparison/`
4. **Performance Metrics**: Add custom evaluation criteria

## Citation

If you use this system in academic work:

```
HFT Regime-Based Statistical Approach Comparison System (2024)
Framework for empirical validation of Frequentist vs Bayesian methods
across market microstructure regimes defined by Liquidity × Volume × Volatility
```

## System Requirements

### Dependencies
- Python 3.8+
- pandas, numpy, scipy
- matplotlib, seaborn
- sklearn for machine learning utilities
- Custom statistical implementations (included)

### Installation
```bash
pip install -r requirements.txt
```

### Testing
```bash
python test_system_import.py
```

## Conclusion

This system transforms your theoretical 9-grid framework into an empirical validation tool, providing concrete evidence for when to use Frequentist vs Bayesian approaches in HFT contexts. The results can guide both academic research and practical trading system development.

**Key Innovation**: Instead of choosing statistical methods based on tradition or convenience, you can now make **data-driven, regime-specific choices** backed by empirical evidence.

---

*Happy regime hunting!*