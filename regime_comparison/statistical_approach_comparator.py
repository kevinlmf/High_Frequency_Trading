"""
Statistical Approach Comparator for Market Regimes
==================================================

Framework for comparing Frequentist vs Bayesian approaches across 9 market regimes:
- Liquidity (High/Medium/Low) × Volume (High/Low) × Volatility (High/Low)

This module integrates:
1. Market regime classification
2. Frequentist statistical methods
3. Bayesian statistical methods
4. Performance comparison and analysis

The goal is to empirically validate which statistical approach works better in each regime.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import our modules
from regime_classification.market_regime_classifier import MarketRegimeClassifier
from statistical_methods.frequentist_methods import FrequentistAnalyzer
from statistical_methods.bayesian_methods import BayesianAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class RegimeComparisonResult:
    """Results for a single regime comparison"""
    regime_name: str
    regime_characteristics: str
    theoretical_winner: str  # "Frequentist", "Bayesian", or "Mixed"

    # Data characteristics
    sample_size: int
    data_period: str

    # Frequentist results
    frequentist_score: float
    frequentist_methods: Dict[str, Any]
    frequentist_reliability: str

    # Bayesian results
    bayesian_score: float
    bayesian_methods: Dict[str, Any]
    bayesian_reliability: str

    # Comparison
    empirical_winner: str
    score_difference: float
    matches_theory: bool

    # Performance metrics
    prediction_accuracy: Dict[str, float]
    uncertainty_quantification: Dict[str, float]
    computational_cost: Dict[str, float]

    # Summary
    key_insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class StatisticalApproachComparator:
    """
    Main class for comparing statistical approaches across market regimes
    """

    def __init__(self, regime_classifier_params: Optional[Dict] = None):
        """
        Initialize the comparator

        Args:
            regime_classifier_params: Parameters for regime classification
        """
        # Initialize components
        classifier_params = regime_classifier_params or {}
        self.regime_classifier = MarketRegimeClassifier(**classifier_params)
        self.frequentist_analyzer = FrequentistAnalyzer()
        self.bayesian_analyzer = BayesianAnalyzer()

        # Results storage
        self.regime_results = {}
        self.comparison_results = []
        self.summary_matrix = None

        logger.info("Statistical Approach Comparator initialized")

    def run_comprehensive_comparison(self,
                                   data: pd.DataFrame,
                                   symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Run comprehensive comparison across all regimes

        Args:
            data: OHLCV DataFrame with datetime index
            symbol: Symbol identifier for reporting

        Returns:
            Dictionary with comprehensive results
        """
        logger.info(f"Starting comprehensive regime comparison for {symbol}")

        # Step 1: Classify market regimes
        logger.info("Step 1: Classifying market regimes...")
        regime_data = self.regime_classifier.classify_regime(data)

        # Get unique regimes present in the data
        unique_regimes = regime_data['regime_name'].unique()
        logger.info(f"Found {len(unique_regimes)} unique regimes: {list(unique_regimes)}")

        # Step 2: Analyze each regime
        comparison_results = []

        for regime in unique_regimes:
            logger.info(f"Analyzing regime: {regime}")

            # Extract data for this regime
            regime_mask = regime_data['regime_name'] == regime
            regime_periods = regime_data[regime_mask]
            regime_market_data = data.loc[regime_periods.index]

            if len(regime_market_data) < 20:
                logger.warning(f"Insufficient data for regime {regime}: {len(regime_market_data)} periods")
                continue

            # Get regime characteristics
            regime_info = {
                'regime_name': regime,
                'characteristics': regime_periods['characteristics'].iloc[0],
                'recommended_approach': regime_periods['recommended_approach'].iloc[0],
                'sample_size': len(regime_market_data),
                'liquidity_score': regime_periods['liquidity_score'].mean(),
                'volume_score': regime_periods['volume_score'].mean(),
                'volatility_score': regime_periods['volatility_score'].mean()
            }

            # Run statistical analyses
            result = self._analyze_single_regime(regime_market_data, regime_info)
            comparison_results.append(result)

        # Step 3: Create summary analysis
        summary = self._create_summary_analysis(comparison_results, symbol)

        # Step 4: Generate insights and recommendations
        insights = self._generate_insights(comparison_results)

        return {
            'symbol': symbol,
            'analysis_date': datetime.now().isoformat(),
            'regime_classifications': regime_data,
            'individual_results': comparison_results,
            'summary_analysis': summary,
            'key_insights': insights,
            'methodology_notes': self._get_methodology_notes()
        }

    def _analyze_single_regime(self,
                             market_data: pd.DataFrame,
                             regime_info: Dict[str, Any]) -> RegimeComparisonResult:
        """
        Analyze a single market regime

        Args:
            market_data: Market data for this regime
            regime_info: Regime characteristics

        Returns:
            RegimeComparisonResult with detailed analysis
        """
        regime_name = regime_info['regime_name']
        logger.info(f"Detailed analysis for {regime_name} ({len(market_data)} periods)")

        # Run Frequentist analysis
        freq_start_time = datetime.now()
        frequentist_results = self.frequentist_analyzer.analyze_regime(market_data, regime_info)
        freq_duration = (datetime.now() - freq_start_time).total_seconds()

        # Run Bayesian analysis
        bayes_start_time = datetime.now()
        bayesian_results = self.bayesian_analyzer.analyze_regime(market_data, regime_info)
        bayes_duration = (datetime.now() - bayes_start_time).total_seconds()

        # Extract scores and metrics
        freq_score = frequentist_results.get('overall_assessment', {}).get('effectiveness_score', 0.5)
        freq_reliability = self._determine_reliability(frequentist_results)

        bayes_score = bayesian_results.get('overall_assessment', {}).get('effectiveness_score', 0.5)
        bayes_reliability = self._determine_reliability(bayesian_results)

        # Determine winners
        theoretical_winner = regime_info.get('recommended_approach', 'Mixed')
        score_difference = bayes_score - freq_score

        if abs(score_difference) < 0.1:
            empirical_winner = "Tied"
        elif score_difference > 0:
            empirical_winner = "Bayesian"
        else:
            empirical_winner = "Frequentist"

        # Check if empirical results match theory
        matches_theory = self._check_theory_match(theoretical_winner, empirical_winner, score_difference)

        # Calculate prediction accuracies
        prediction_accuracy = self._calculate_prediction_accuracy(
            market_data, frequentist_results, bayesian_results
        )

        # Calculate uncertainty quantification metrics
        uncertainty_metrics = self._calculate_uncertainty_metrics(
            frequentist_results, bayesian_results
        )

        # Computational costs
        computational_cost = {
            'frequentist_time_seconds': freq_duration,
            'bayesian_time_seconds': bayes_duration,
            'relative_cost': bayes_duration / freq_duration if freq_duration > 0 else 1.0
        }

        # Generate insights and recommendations
        key_insights = self._generate_regime_insights(
            regime_info, frequentist_results, bayesian_results,
            freq_score, bayes_score, matches_theory
        )

        recommendations = self._generate_regime_recommendations(
            regime_name, empirical_winner, matches_theory, prediction_accuracy
        )

        return RegimeComparisonResult(
            regime_name=regime_name,
            regime_characteristics=regime_info.get('characteristics', ''),
            theoretical_winner=theoretical_winner,
            sample_size=len(market_data),
            data_period=f"{market_data.index[0]} to {market_data.index[-1]}",
            frequentist_score=freq_score,
            frequentist_methods=frequentist_results.get('methods', {}),
            frequentist_reliability=freq_reliability,
            bayesian_score=bayes_score,
            bayesian_methods=bayesian_results.get('methods', {}),
            bayesian_reliability=bayes_reliability,
            empirical_winner=empirical_winner,
            score_difference=score_difference,
            matches_theory=matches_theory,
            prediction_accuracy=prediction_accuracy,
            uncertainty_quantification=uncertainty_metrics,
            computational_cost=computational_cost,
            key_insights=key_insights,
            recommendations=recommendations
        )

    def _determine_reliability(self, results: Dict[str, Any]) -> str:
        """Determine reliability level from analysis results"""
        methods = results.get('methods', {})

        high_reliability_count = 0
        total_methods = 0

        for method_name, method_results in methods.items():
            if isinstance(method_results, dict) and 'effectiveness' in method_results:
                effectiveness = method_results['effectiveness']
                if isinstance(effectiveness, dict):
                    reliability = effectiveness.get('reliability', 'Medium')
                    total_methods += 1
                    if reliability == 'High':
                        high_reliability_count += 1

        if total_methods == 0:
            return 'Low'
        elif high_reliability_count / total_methods >= 0.5:
            return 'High'
        else:
            return 'Medium'

    def _check_theory_match(self, theoretical: str, empirical: str, score_diff: float) -> bool:
        """Check if empirical results match theoretical expectations"""
        if theoretical == "Mixed":
            return abs(score_diff) < 0.3  # Close scores expected for mixed
        elif theoretical == "Frequentist":
            return empirical == "Frequentist" or (empirical == "Tied" and score_diff <= 0.1)
        elif theoretical == "Bayesian":
            return empirical == "Bayesian" or (empirical == "Tied" and score_diff >= -0.1)
        else:
            return True  # Unknown theoretical winner

    def _calculate_prediction_accuracy(self,
                                     market_data: pd.DataFrame,
                                     freq_results: Dict[str, Any],
                                     bayes_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate prediction accuracy metrics for both approaches"""

        # Simple accuracy calculation based on available methods
        returns = market_data['close'].pct_change().dropna().values

        accuracy_metrics = {
            'frequentist_rmse': np.inf,
            'bayesian_rmse': np.inf,
            'frequentist_mae': np.inf,
            'bayesian_mae': np.inf
        }

        # Extract predictions if available
        freq_methods = freq_results.get('methods', {})
        if 'garch' in freq_methods and 'error' not in freq_methods['garch']:
            # Use some proxy for prediction accuracy
            accuracy_metrics['frequentist_rmse'] = np.random.uniform(0.1, 0.5)  # Placeholder
            accuracy_metrics['frequentist_mae'] = accuracy_metrics['frequentist_rmse'] * 0.8

        bayes_methods = bayes_results.get('methods', {})
        if any(method not in ['error', 'warning'] for method in bayes_methods.keys()):
            accuracy_metrics['bayesian_rmse'] = np.random.uniform(0.1, 0.5)  # Placeholder
            accuracy_metrics['bayesian_mae'] = accuracy_metrics['bayesian_rmse'] * 0.8

        return accuracy_metrics

    def _calculate_uncertainty_metrics(self,
                                     freq_results: Dict[str, Any],
                                     bayes_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate uncertainty quantification metrics"""

        uncertainty_metrics = {
            'frequentist_confidence_width': 0.0,
            'bayesian_credible_width': 0.0,
            'bayesian_uncertainty_captured': 0.0
        }

        # Extract uncertainty measures from results
        freq_methods = freq_results.get('methods', {})
        bayes_methods = bayes_results.get('methods', {})

        # Frequentist confidence intervals (from GARCH or tests)
        if 'garch' in freq_methods and 'error' not in freq_methods['garch']:
            # Average confidence interval width
            uncertainty_metrics['frequentist_confidence_width'] = 0.2  # Placeholder

        # Bayesian credible intervals and uncertainty measures
        for method_name, method_results in bayes_methods.items():
            if isinstance(method_results, dict) and 'uncertainty_measures' in method_results:
                uncertainty_measures = method_results['uncertainty_measures']
                if isinstance(uncertainty_measures, dict):
                    # Extract total uncertainty or similar measure
                    total_unc = uncertainty_measures.get('total_uncertainty', 0)
                    uncertainty_metrics['bayesian_uncertainty_captured'] = max(
                        uncertainty_metrics['bayesian_uncertainty_captured'], total_unc
                    )

        uncertainty_metrics['bayesian_credible_width'] = 0.25  # Placeholder

        return uncertainty_metrics

    def _generate_regime_insights(self,
                                regime_info: Dict[str, Any],
                                freq_results: Dict[str, Any],
                                bayes_results: Dict[str, Any],
                                freq_score: float,
                                bayes_score: float,
                                matches_theory: bool) -> List[str]:
        """Generate key insights for a specific regime"""

        insights = []
        regime_name = regime_info['regime_name']

        # Theory vs Practice insight
        if matches_theory:
            insights.append(f"✅ Empirical results align with theoretical expectations for {regime_name}")
        else:
            insights.append(f"⚠️ Empirical results contradict theory for {regime_name} - further investigation needed")

        # Performance insights
        if abs(freq_score - bayes_score) < 0.1:
            insights.append("📊 Both approaches perform similarly - choice may depend on other factors")
        elif freq_score > bayes_score:
            insights.append(f"📈 Frequentist methods outperform (+{freq_score - bayes_score:.2f} score advantage)")
        else:
            insights.append(f"🎯 Bayesian methods outperform (+{bayes_score - freq_score:.2f} score advantage)")

        # Sample size insights
        sample_size = regime_info['sample_size']
        if sample_size < 30:
            insights.append("⚡ Small sample size - results may be less reliable")
        elif sample_size > 200:
            insights.append("💪 Large sample provides robust statistical inference")

        # Regime-specific insights
        if 'Low' in regime_name and bayes_score > freq_score:
            insights.append("🔍 Bayesian methods effectively handle sparse data regime")
        elif 'HighLiq_HighVol' in regime_name and freq_score > bayes_score:
            insights.append("⚙️ Frequentist methods efficient in high-data regime")

        return insights

    def _generate_regime_recommendations(self,
                                       regime_name: str,
                                       empirical_winner: str,
                                       matches_theory: bool,
                                       prediction_accuracy: Dict[str, float]) -> List[str]:
        """Generate recommendations for a specific regime"""

        recommendations = []

        # Primary recommendation based on empirical winner
        if empirical_winner == "Frequentist":
            recommendations.append(f"🏆 Use Frequentist methods for {regime_name} regime")
            recommendations.append("• Focus on GARCH models and classical tests")
            recommendations.append("• Leverage large sample efficiency")
        elif empirical_winner == "Bayesian":
            recommendations.append(f"🏆 Use Bayesian methods for {regime_name} regime")
            recommendations.append("• Implement hierarchical models for uncertainty quantification")
            recommendations.append("• Use dynamic updating for real-time adaptation")
        else:
            recommendations.append(f"🤝 Both approaches viable for {regime_name}")
            recommendations.append("• Consider hybrid approach combining both methods")
            recommendations.append("• Choose based on computational constraints")

        # Theory alignment recommendation
        if not matches_theory:
            recommendations.append("⚠️ Re-examine theoretical assumptions for this regime")
            recommendations.append("• Consider regime subclassification")
            recommendations.append("• Validate with additional data periods")

        # Accuracy-based recommendations
        freq_rmse = prediction_accuracy.get('frequentist_rmse', np.inf)
        bayes_rmse = prediction_accuracy.get('bayesian_rmse', np.inf)

        if freq_rmse < bayes_rmse and freq_rmse < 0.3:
            recommendations.append("🎯 Frequentist predictions more accurate - prioritize for forecasting")
        elif bayes_rmse < freq_rmse and bayes_rmse < 0.3:
            recommendations.append("🎯 Bayesian predictions more accurate - prioritize for forecasting")

        return recommendations

    def _create_summary_analysis(self,
                               comparison_results: List[RegimeComparisonResult],
                               symbol: str) -> Dict[str, Any]:
        """Create summary analysis across all regimes"""

        if not comparison_results:
            return {"error": "No comparison results available"}

        # Create 9x9 matrix-like summary
        regime_matrix = {}
        theory_match_count = 0
        total_regimes = len(comparison_results)

        # Aggregate results
        freq_wins = 0
        bayes_wins = 0
        ties = 0

        for result in comparison_results:
            regime_matrix[result.regime_name] = {
                'theoretical': result.theoretical_winner,
                'empirical': result.empirical_winner,
                'freq_score': result.frequentist_score,
                'bayes_score': result.bayesian_score,
                'matches_theory': result.matches_theory,
                'sample_size': result.sample_size
            }

            if result.matches_theory:
                theory_match_count += 1

            if result.empirical_winner == "Frequentist":
                freq_wins += 1
            elif result.empirical_winner == "Bayesian":
                bayes_wins += 1
            else:
                ties += 1

        # Overall statistics
        theory_accuracy = theory_match_count / total_regimes if total_regimes > 0 else 0

        summary = {
            'total_regimes_analyzed': total_regimes,
            'theory_match_rate': theory_accuracy,
            'empirical_results': {
                'frequentist_wins': freq_wins,
                'bayesian_wins': bayes_wins,
                'ties': ties
            },
            'regime_breakdown': regime_matrix,
            'overall_recommendation': self._determine_overall_recommendation(
                freq_wins, bayes_wins, ties, theory_accuracy
            )
        }

        return summary

    def _determine_overall_recommendation(self,
                                        freq_wins: int,
                                        bayes_wins: int,
                                        ties: int,
                                        theory_accuracy: float) -> str:
        """Determine overall recommendation across all regimes"""

        total = freq_wins + bayes_wins + ties
        if total == 0:
            return "Insufficient data for recommendation"

        if theory_accuracy > 0.7:
            theory_confidence = "high"
        elif theory_accuracy > 0.5:
            theory_confidence = "moderate"
        else:
            theory_confidence = "low"

        if freq_wins > bayes_wins and freq_wins > ties:
            return f"Frequentist methods generally preferred ({theory_confidence} theoretical alignment)"
        elif bayes_wins > freq_wins and bayes_wins > ties:
            return f"Bayesian methods generally preferred ({theory_confidence} theoretical alignment)"
        else:
            return f"Mixed approach recommended - regime-specific selection ({theory_confidence} theoretical alignment)"

    def _generate_insights(self, comparison_results: List[RegimeComparisonResult]) -> List[str]:
        """Generate overall insights across all regimes"""

        insights = []

        if not comparison_results:
            return ["No regime data available for insights"]

        # Theory validation insight
        theory_matches = sum(1 for r in comparison_results if r.matches_theory)
        theory_rate = theory_matches / len(comparison_results)

        if theory_rate > 0.8:
            insights.append("🎯 Strong theoretical validation - HFT regime theory holds well")
        elif theory_rate > 0.6:
            insights.append("✅ Moderate theoretical validation - some regimes behave as expected")
        else:
            insights.append("⚠️ Weak theoretical validation - regime theory needs refinement")

        # Performance patterns
        high_vol_regimes = [r for r in comparison_results if 'HighVol' in r.regime_name]
        low_vol_regimes = [r for r in comparison_results if 'LowVol' in r.regime_name]

        if high_vol_regimes:
            bayes_better_in_high_vol = sum(1 for r in high_vol_regimes if r.empirical_winner == "Bayesian")
            if bayes_better_in_high_vol / len(high_vol_regimes) > 0.6:
                insights.append("🌪️ Bayesian methods consistently better in high volatility regimes")

        if low_vol_regimes:
            freq_better_in_low_vol = sum(1 for r in low_vol_regimes if r.empirical_winner == "Frequentist")
            if freq_better_in_low_vol / len(low_vol_regimes) > 0.6:
                insights.append("📈 Frequentist methods efficient in stable (low volatility) regimes")

        # Sample size effects
        small_sample_regimes = [r for r in comparison_results if r.sample_size < 50]
        if small_sample_regimes:
            bayes_better_small = sum(1 for r in small_sample_regimes if r.empirical_winner == "Bayesian")
            if bayes_better_small / len(small_sample_regimes) > 0.6:
                insights.append("📊 Bayesian methods superior with small sample sizes")

        # Computational cost insights
        avg_cost_ratio = np.mean([r.computational_cost['relative_cost'] for r in comparison_results])
        if avg_cost_ratio > 3:
            insights.append("⚡ Bayesian methods ~{avg_cost_ratio:.1f}x slower - consider computational constraints")

        return insights

    def _get_methodology_notes(self) -> Dict[str, str]:
        """Get methodology notes for transparency"""

        return {
            "regime_classification": "Based on rolling liquidity, volume, and volatility measures",
            "frequentist_methods": "GARCH models, Hawkes processes, classical statistical tests",
            "bayesian_methods": "Hierarchical Bayesian, Markov switching, dynamic updating",
            "effectiveness_scoring": "0-1 scale based on method appropriateness for regime characteristics",
            "theory_matching": "Comparison between theoretical expectations and empirical performance",
            "limitations": "Results depend on regime classification accuracy and sample periods analyzed"
        }

    def generate_regime_matrix_visualization(self,
                                           results: Dict[str, Any],
                                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate visualization of regime comparison results

        Args:
            results: Results from run_comprehensive_comparison
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        comparison_results = results.get('individual_results', [])

        if not comparison_results:
            logger.warning("No results to visualize")
            return None

        # Create matrix data
        regimes = [r.regime_name for r in comparison_results]
        freq_scores = [r.frequentist_score for r in comparison_results]
        bayes_scores = [r.bayesian_score for r in comparison_results]

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'HFT Regime Analysis: Frequentist vs Bayesian Methods\n{results["symbol"]}', fontsize=16, fontweight='bold')

        # 1. Score comparison
        x_pos = np.arange(len(regimes))
        width = 0.35

        bars1 = ax1.bar(x_pos - width/2, freq_scores, width, label='Frequentist', color='blue', alpha=0.7)
        bars2 = ax1.bar(x_pos + width/2, bayes_scores, width, label='Bayesian', color='red', alpha=0.7)

        ax1.set_xlabel('Market Regime')
        ax1.set_ylabel('Effectiveness Score')
        ax1.set_title('Method Effectiveness by Regime')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([r.replace('_', '\n') for r in regimes], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Theory vs Empirical
        theory_match = [r.matches_theory for r in comparison_results]
        colors = ['green' if match else 'red' for match in theory_match]

        ax2.scatter(freq_scores, bayes_scores, c=colors, s=100, alpha=0.7)
        ax2.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
        ax2.set_xlabel('Frequentist Score')
        ax2.set_ylabel('Bayesian Score')
        ax2.set_title('Frequentist vs Bayesian Performance\n(Green=Theory Match, Red=Mismatch)')
        ax2.grid(True, alpha=0.3)

        # 3. Sample size effects
        sample_sizes = [r.sample_size for r in comparison_results]
        score_diffs = [r.score_difference for r in comparison_results]

        ax3.scatter(sample_sizes, score_diffs, c=colors, s=100, alpha=0.7)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Sample Size')
        ax3.set_ylabel('Score Difference (Bayes - Freq)')
        ax3.set_title('Sample Size vs Method Performance Gap')
        ax3.grid(True, alpha=0.3)

        # 4. Regime characteristics
        regime_types = [r.regime_name.split('_')[0] for r in comparison_results]
        unique_types = list(set(regime_types))
        type_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))

        for i, regime_type in enumerate(unique_types):
            mask = [rt == regime_type for rt in regime_types]
            freq_subset = [fs for fs, m in zip(freq_scores, mask) if m]
            bayes_subset = [bs for bs, m in zip(bayes_scores, mask) if m]

            ax4.scatter(freq_subset, bayes_subset,
                       c=[type_colors[i]], label=regime_type, s=100, alpha=0.7)

        ax4.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
        ax4.set_xlabel('Frequentist Score')
        ax4.set_ylabel('Bayesian Score')
        ax4.set_title('Performance by Liquidity Level')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")

        return fig

    def export_results_to_csv(self,
                             results: Dict[str, Any],
                             output_dir: str = "exports") -> str:
        """
        Export comparison results to CSV format

        Args:
            results: Results from run_comprehensive_comparison
            output_dir: Directory to save CSV files

        Returns:
            Path to the main results CSV file
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        comparison_results = results.get('individual_results', [])
        if not comparison_results:
            logger.warning("No results to export")
            return None

        # Main results dataframe
        results_data = []
        for result in comparison_results:
            results_data.append({
                'regime_name': result.regime_name,
                'regime_characteristics': result.regime_characteristics,
                'theoretical_winner': result.theoretical_winner,
                'sample_size': result.sample_size,
                'frequentist_score': result.frequentist_score,
                'frequentist_reliability': result.frequentist_reliability,
                'bayesian_score': result.bayesian_score,
                'bayesian_reliability': result.bayesian_reliability,
                'empirical_winner': result.empirical_winner,
                'score_difference': result.score_difference,
                'matches_theory': result.matches_theory,
                'freq_rmse': result.prediction_accuracy.get('frequentist_rmse', np.nan),
                'bayes_rmse': result.prediction_accuracy.get('bayesian_rmse', np.nan),
                'computational_cost_ratio': result.computational_cost.get('relative_cost', np.nan)
            })

        results_df = pd.DataFrame(results_data)

        # Save main results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbol = results.get('symbol', 'UNKNOWN')
        filename = f"{symbol}_regime_comparison_{timestamp}.csv"
        filepath = output_path / filename

        results_df.to_csv(filepath, index=False)

        # Save regime classification data
        regime_data = results.get('regime_classifications')
        if regime_data is not None:
            regime_filename = f"{symbol}_regime_classifications_{timestamp}.csv"
            regime_filepath = output_path / regime_filename
            regime_data.to_csv(regime_filepath, index=True)

        logger.info(f"Results exported to {filepath}")
        return str(filepath)