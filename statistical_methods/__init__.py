"""
Statistical Methods Module
==========================

Statistical methods for HFT trading analysis:
- Frequentist methods: GARCH, Hawkes processes, classical tests
- Bayesian methods: Hierarchical Bayesian, state switching, dynamic priors

These methods are evaluated across different market regimes to determine optimal approaches.
"""

from .frequentist_methods import (
    FrequentistMethod,
    FrequentistResult,
    GARCHModel,
    HawkesProcess,
    ClassicalTests,
    FrequentistAnalyzer
)

from .bayesian_methods import (
    BayesianMethod,
    BayesianResult,
    HierarchicalBayesianModel,
    MarkovSwitchingModel,
    DynamicBayesianUpdater,
    BayesianAnalyzer
)

__all__ = [
    'FrequentistMethod',
    'FrequentistResult',
    'GARCHModel',
    'HawkesProcess',
    'ClassicalTests',
    'FrequentistAnalyzer',
    'BayesianMethod',
    'BayesianResult',
    'HierarchicalBayesianModel',
    'MarkovSwitchingModel',
    'DynamicBayesianUpdater',
    'BayesianAnalyzer'
]