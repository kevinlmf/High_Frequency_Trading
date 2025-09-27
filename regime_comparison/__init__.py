"""
Regime Comparison Module
=======================

Framework for comparing statistical approaches across different market regimes.
Integrates regime classification with Frequentist and Bayesian method evaluation.
"""

from .statistical_approach_comparator import (
    StatisticalApproachComparator,
    RegimeComparisonResult
)

__all__ = [
    'StatisticalApproachComparator',
    'RegimeComparisonResult'
]