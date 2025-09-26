"""
Strategy Comparison PDF Report Generator

Generates comprehensive PDF reports comparing multiple HFT trading strategies
with performance metrics, risk analysis, and visual comparisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import warnings

# Set style for professional charts
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class StrategyComparisonPDFGenerator:
    """
    Multi-strategy comparison PDF report generator

    Creates comprehensive PDF reports comparing HFT strategies across multiple metrics
    including net PnL, risk-adjusted returns, drawdowns, and cost analysis.
    """

    def __init__(self, report_title: str = "HFT Strategy Comparison Report"):
        self.report_title = report_title
        self.fig_size = (12, 8)
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#8E44AD']

        # Set font configuration for charts
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def generate_strategy_comparison_pdf(self,
                                       strategies_results: Dict[str, Dict],
                                       save_path: str = "strategy_comparison_report.pdf") -> str:
        """
        Generate strategy comparison PDF report

        Args:
            strategies_results: Strategy results dictionary {strategy_name: {performance_metrics, returns_series, ...}}
            save_path: PDF save path

        Returns:
            Path of the generated PDF file
        """

        with PdfPages(save_path) as pdf:
            # Page 1: Cover and strategy overview
            self._create_cover_page(pdf, strategies_results)

            # Page 2: Net PnL comparison
            self._create_net_pnl_comparison(pdf, strategies_results)

            # Page 3: Risk-return scatter plot
            self._create_risk_return_scatter(pdf, strategies_results)

            # Page 4: Strategy performance ranking
            self._create_strategy_ranking(pdf, strategies_results)

            # Page 5: Cost analysis comparison
            self._create_cost_comparison(pdf, strategies_results)

            # Page 6: Drawdown comparison
            self._create_drawdown_comparison(pdf, strategies_results)

            # Page 7: Detailed metrics comparison table
            self._create_detailed_comparison_table(pdf, strategies_results)

            # Page 8: Recommendations summary
            self._create_recommendations(pdf, strategies_results)

        print(f"Strategy comparison PDF report generated: {save_path}")
        return save_path

    def _create_cover_page(self, pdf: PdfPages, strategies_results: Dict[str, Dict]):
        """Create cover page with strategy overview"""
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.axis('off')

        # Title
        ax.text(0.5, 0.9, self.report_title, fontsize=24, fontweight='bold',
                ha='center', transform=ax.transAxes)

        # Generation time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(0.5, 0.85, f"Generated: {current_time}", fontsize=12,
                ha='center', transform=ax.transAxes)

        # Strategy overview
        ax.text(0.5, 0.75, "Strategy Overview", fontsize=18, fontweight='bold',
                ha='center', transform=ax.transAxes)

        # Strategy list and core metrics
        y_pos = 0.65
        ax.text(0.1, y_pos, "Strategies Analyzed:", fontsize=14, fontweight='bold',
                transform=ax.transAxes)
        y_pos -= 0.05

        for i, (strategy_name, results) in enumerate(strategies_results.items()):
            metrics = results.get('performance_metrics', {})
            net_pnl = metrics.get('net_total_pnl', 0)
            sharpe = metrics.get('sharpe_ratio', 0)

            color = self.colors[i % len(self.colors)]
            ax.text(0.15, y_pos, f"• {strategy_name}", fontsize=12,
                   color=color, fontweight='bold', transform=ax.transAxes)
            ax.text(0.6, y_pos, f"Net PnL: ${net_pnl:,.0f}, Sharpe: {sharpe:.2f}",
                   fontsize=11, transform=ax.transAxes)
            y_pos -= 0.04

        # Best strategy
        if strategies_results:
            best_strategy = max(strategies_results.items(),
                              key=lambda x: x[1].get('performance_metrics', {}).get('sharpe_ratio', -999))
            best_name = best_strategy[0]
            best_metrics = best_strategy[1].get('performance_metrics', {})

            y_pos -= 0.05
            ax.text(0.5, y_pos, f"Best Strategy: {best_name}", fontsize=16, fontweight='bold',
                    ha='center', transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.5))

        # Analysis description
        ax.text(0.5, 0.25, "This report compares multiple HFT strategies based on:\n" +
                "• Net PnL after all transaction costs\n" +
                "• Risk-adjusted returns (Sharpe ratio)\n" +
                "• Maximum drawdown and volatility\n" +
                "• Cost efficiency and breakeven analysis",
                fontsize=12, ha='center', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_net_pnl_comparison(self, pdf: PdfPages, strategies_results: Dict[str, Dict]):
        """Create net PnL comparison charts"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.fig_size, height_ratios=[3, 1])

        # Main chart: Cumulative net PnL curves
        for i, (strategy_name, results) in enumerate(strategies_results.items()):
            returns_series = results.get('returns_series')
            if returns_series is not None:
                # Calculate cumulative net PnL
                cumulative_returns = (1 + returns_series).cumprod() - 1

                color = self.colors[i % len(self.colors)]
                ax1.plot(cumulative_returns.index, cumulative_returns * 100,
                        color=color, linewidth=2.5, label=strategy_name, alpha=0.8)

        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_title('Cumulative Net Returns Comparison', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Sub-chart: Final net PnL bar chart
        strategy_names = list(strategies_results.keys())
        final_pnls = []

        for strategy_name in strategy_names:
            metrics = strategies_results[strategy_name].get('performance_metrics', {})
            final_pnls.append(metrics.get('net_total_pnl', 0))

        bars = ax2.bar(range(len(strategy_names)), final_pnls,
                      color=self.colors[:len(strategy_names)], alpha=0.7, edgecolor='black')

        # Add value labels
        for bar, pnl in zip(bars, final_pnls):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (abs(height) * 0.01),
                    f'${pnl:,.0f}', ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold')

        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Final Net PnL Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Net PnL ($)', fontsize=12)
        ax2.set_xticks(range(len(strategy_names)))
        ax2.set_xticklabels(strategy_names, rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_risk_return_scatter(self, pdf: PdfPages, strategies_results: Dict[str, Dict]):
        """Create risk-return scatter plot"""
        fig, ax = plt.subplots(figsize=self.fig_size)

        returns_list = []
        volatility_list = []
        sharpe_list = []
        names_list = []
        colors_list = []

        for i, (strategy_name, results) in enumerate(strategies_results.items()):
            metrics = results.get('performance_metrics', {})

            annual_return = metrics.get('annualized_return', 0) * 100
            volatility = metrics.get('volatility', 0) * 100
            sharpe = metrics.get('sharpe_ratio', 0)

            returns_list.append(annual_return)
            volatility_list.append(volatility)
            sharpe_list.append(sharpe)
            names_list.append(strategy_name)
            colors_list.append(self.colors[i % len(self.colors)])

        # Create scatter plot, point size represents Sharpe ratio
        sizes = [max(50, abs(s) * 100 + 50) for s in sharpe_list]
        scatter = ax.scatter(volatility_list, returns_list, s=sizes, c=colors_list,
                           alpha=0.7, edgecolors='black', linewidth=2)

        # Add strategy name labels
        for i, name in enumerate(names_list):
            ax.annotate(name, (volatility_list[i], returns_list[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=10,
                       fontweight='bold')

        # Add quadrant lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=np.mean(volatility_list), color='gray', linestyle='--', alpha=0.5)

        # Label quadrants
        ax.text(0.02, 0.98, 'Low Risk\nHigh Return', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.5),
                verticalalignment='top')
        ax.text(0.75, 0.02, 'High Risk\nLow Return', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.5),
                verticalalignment='bottom')

        ax.set_xlabel('Annualized Volatility (%)', fontsize=12)
        ax.set_ylabel('Annualized Return (%)', fontsize=12)
        ax.set_title('Risk-Return Analysis\n(Bubble size = Sharpe ratio)',
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_strategy_ranking(self, pdf: PdfPages, strategies_results: Dict[str, Dict]):
        """Create strategy ranking page"""
        fig = plt.figure(figsize=self.fig_size)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Sharpe ratio ranking
        ax1 = fig.add_subplot(gs[0, 0])
        sharpe_data = [(name, results['performance_metrics'].get('sharpe_ratio', 0))
                      for name, results in strategies_results.items()]
        sharpe_data.sort(key=lambda x: x[1], reverse=True)

        names, values = zip(*sharpe_data)
        bars = ax1.barh(range(len(names)), values, color=self.colors[:len(names)], alpha=0.7)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names)
        ax1.set_xlabel('Sharpe Ratio')
        ax1.set_title('Sharpe Ratio Ranking', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for bar, value in zip(bars, values):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', va='center', fontweight='bold')

        # 2. Net PnL ranking
        ax2 = fig.add_subplot(gs[0, 1])
        pnl_data = [(name, results['performance_metrics'].get('net_total_pnl', 0))
                   for name, results in strategies_results.items()]
        pnl_data.sort(key=lambda x: x[1], reverse=True)

        names, values = zip(*pnl_data)
        bars = ax2.barh(range(len(names)), values, color=self.colors[:len(names)], alpha=0.7)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names)
        ax2.set_xlabel('Net PnL ($)')
        ax2.set_title('Net PnL Ranking', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for bar, value in zip(bars, values):
            ax2.text(bar.get_width() + abs(max(values)) * 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'${value:,.0f}', va='center', fontweight='bold')

        # 3. Maximum drawdown ranking (lower is better)
        ax3 = fig.add_subplot(gs[1, 0])
        drawdown_data = [(name, abs(results['performance_metrics'].get('max_drawdown', 0)) * 100)
                        for name, results in strategies_results.items()]
        drawdown_data.sort(key=lambda x: x[1])  # Ascending order, lower drawdown is better

        names, values = zip(*drawdown_data)
        bars = ax3.barh(range(len(names)), values, color=self.colors[:len(names)], alpha=0.7)
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels(names)
        ax3.set_xlabel('Max Drawdown (%)')
        ax3.set_title('Max Drawdown Ranking\n(Lower is Better)', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for bar, value in zip(bars, values):
            ax3.text(bar.get_width() + max(values) * 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{value:.2f}%', va='center', fontweight='bold')

        # 4. Composite score ranking
        ax4 = fig.add_subplot(gs[1, 1])

        # Calculate composite score (weights can be adjusted as needed)
        composite_scores = []
        for name, results in strategies_results.items():
            metrics = results['performance_metrics']

            # Normalize each metric (0-100 score)
            sharpe_score = max(0, min(100, metrics.get('sharpe_ratio', 0) * 25 + 50))
            pnl_score = max(0, min(100, (metrics.get('cumulative_return', 0) * 100 + 10)))
            drawdown_score = max(0, min(100, 100 - abs(metrics.get('max_drawdown', 0)) * 500))

            composite = (sharpe_score * 0.4 + pnl_score * 0.4 + drawdown_score * 0.2)
            composite_scores.append((name, composite))

        composite_scores.sort(key=lambda x: x[1], reverse=True)

        names, values = zip(*composite_scores)
        bars = ax4.barh(range(len(names)), values, color=self.colors[:len(names)], alpha=0.7)
        ax4.set_yticks(range(len(names)))
        ax4.set_yticklabels(names)
        ax4.set_xlabel('Composite Score')
        ax4.set_title('Overall Ranking', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for bar, value in zip(bars, values):
            ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{value:.0f}', va='center', fontweight='bold')

        plt.suptitle('Strategy Performance Rankings', fontsize=16, fontweight='bold', y=0.98)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_cost_comparison(self, pdf: PdfPages, strategies_results: Dict[str, Dict]):
        """Create cost comparison analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.fig_size)

        strategy_names = list(strategies_results.keys())

        # 1. Cost drag comparison
        cost_drags = []
        for name in strategy_names:
            metrics = strategies_results[name]['performance_metrics']
            cost_drags.append(metrics.get('cost_drag_pct', 0))

        ax1.bar(range(len(strategy_names)), cost_drags,
               color=self.colors[:len(strategy_names)], alpha=0.7)
        ax1.set_title('Cost Drag Comparison', fontweight='bold')
        ax1.set_ylabel('Cost Drag (%)')
        ax1.set_xticks(range(len(strategy_names)))
        ax1.set_xticklabels(strategy_names, rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Net profit margin comparison
        profit_margins = []
        for name in strategy_names:
            metrics = strategies_results[name]['performance_metrics']
            profit_margins.append(metrics.get('net_profit_margin', 0))

        bars = ax2.bar(range(len(strategy_names)), profit_margins,
                      color=self.colors[:len(strategy_names)], alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Net Profit Margin', fontweight='bold')
        ax2.set_ylabel('Profit Margin (%)')
        ax2.set_xticks(range(len(strategy_names)))
        ax2.set_xticklabels(strategy_names, rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Return to cost ratio comparison
        return_cost_ratios = []
        for name in strategy_names:
            metrics = strategies_results[name]['performance_metrics']
            return_cost_ratios.append(metrics.get('return_to_cost_ratio', 0))

        ax3.bar(range(len(strategy_names)), return_cost_ratios,
               color=self.colors[:len(strategy_names)], alpha=0.7)
        ax3.set_title('Return to Cost Ratio', fontweight='bold')
        ax3.set_ylabel('Ratio')
        ax3.set_xticks(range(len(strategy_names)))
        ax3.set_xticklabels(strategy_names, rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Cost efficiency summary table
        ax4.axis('off')

        # Create cost efficiency table data
        table_data = []
        for name in strategy_names:
            metrics = strategies_results[name]['performance_metrics']
            table_data.append([
                name[:12] + '...' if len(name) > 12 else name,
                f"{metrics.get('cost_drag_pct', 0):.1f}%",
                f"{metrics.get('net_profit_margin', 0):.1f}%",
                f"{metrics.get('return_to_cost_ratio', 0):.1f}"
            ])

        table = ax4.table(cellText=table_data,
                         colLabels=['Strategy', 'Cost Drag', 'Profit Margin', 'Return/Cost'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Set table style
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#E6E6FA')
            else:
                cell.set_facecolor('#F8F8FF')

        ax4.set_title('Cost Efficiency Summary', fontsize=14, fontweight='bold', pad=20)

        plt.suptitle('Cost Analysis Comparison', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_drawdown_comparison(self, pdf: PdfPages, strategies_results: Dict[str, Dict]):
        """Create drawdown comparison"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.fig_size)

        # 1. Drawdown curve comparison
        drawdown = None  # Initialize drawdown
        for i, (strategy_name, results) in enumerate(strategies_results.items()):
            returns_series = results.get('returns_series')
            if returns_series is not None:
                # Calculate drawdown
                cumulative = (1 + returns_series).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max * 100

                color = self.colors[i % len(self.colors)]
                ax1.plot(drawdown.index, drawdown, color=color, linewidth=2,
                        label=strategy_name, alpha=0.8)

        if drawdown is not None:
            ax1.fill_between(drawdown.index, 0, drawdown, alpha=0.3)
        ax1.set_title('Drawdown Comparison Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Drawdown (%)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # 2. Maximum drawdown comparison bar chart
        strategy_names = list(strategies_results.keys())
        max_drawdowns = []

        for name in strategy_names:
            metrics = strategies_results[name]['performance_metrics']
            max_drawdowns.append(abs(metrics.get('max_drawdown', 0)) * 100)

        bars = ax2.bar(range(len(strategy_names)), max_drawdowns,
                      color=self.colors[:len(strategy_names)], alpha=0.7, edgecolor='black')

        # Add value labels
        for bar, dd in zip(bars, max_drawdowns):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{dd:.2f}%', ha='center', va='bottom', fontweight='bold')

        ax2.set_title('Maximum Drawdown Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Max Drawdown (%)')
        ax2.set_xticks(range(len(strategy_names)))
        ax2.set_xticklabels(strategy_names, rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_detailed_comparison_table(self, pdf: PdfPages, strategies_results: Dict[str, Dict]):
        """Create detailed comparison table"""
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.axis('off')

        # Collect key metrics for all strategies
        table_data = []
        for strategy_name, results in strategies_results.items():
            metrics = results['performance_metrics']

            row = [
                strategy_name[:15] + '...' if len(strategy_name) > 15 else strategy_name,
                f"${metrics.get('net_total_pnl', 0):,.0f}",
                f"{metrics.get('cumulative_return', 0) * 100:.2f}%",
                f"{metrics.get('sharpe_ratio', 0):.3f}",
                f"{abs(metrics.get('max_drawdown', 0)) * 100:.2f}%",
                f"{metrics.get('volatility', 0) * 100:.2f}%",
                f"{metrics.get('win_rate', 0) * 100:.1f}%",
                f"{metrics.get('cost_drag_pct', 0):.1f}%"
            ]
            table_data.append(row)

        # Create table
        table = ax.table(
            cellText=table_data,
            colLabels=['Strategy', 'Net PnL', 'Return', 'Sharpe', 'Max DD', 'Vol', 'Win Rate', 'Cost Drag'],
            cellLoc='center',
            loc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)

        # Set table style
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(color='white')
            else:
                # Set color based on strategy performance
                if j == 1:  # Net PnL column
                    value = float(table_data[i-1][j].replace('$', '').replace(',', ''))
                    if value > 0:
                        cell.set_facecolor('#E8F5E8')
                    else:
                        cell.set_facecolor('#FFF0F0')
                elif j == 3:  # Sharpe column
                    value = float(table_data[i-1][j])
                    if value > 1.0:
                        cell.set_facecolor('#E8F5E8')
                    elif value > 0:
                        cell.set_facecolor('#FFF9E6')
                    else:
                        cell.set_facecolor('#FFF0F0')
                else:
                    cell.set_facecolor('#F8F8FF')

        ax.set_title('Detailed Strategy Comparison', fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_recommendations(self, pdf: PdfPages, strategies_results: Dict[str, Dict]):
        """Create recommendations summary page"""
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.axis('off')

        ax.text(0.5, 0.95, 'Strategy Recommendations', fontsize=20, fontweight='bold',
                ha='center', transform=ax.transAxes)

        # Find best strategies
        best_sharpe = max(strategies_results.items(),
                         key=lambda x: x[1]['performance_metrics'].get('sharpe_ratio', -999))
        best_pnl = max(strategies_results.items(),
                      key=lambda x: x[1]['performance_metrics'].get('net_total_pnl', -999999))
        best_drawdown = min(strategies_results.items(),
                           key=lambda x: abs(x[1]['performance_metrics'].get('max_drawdown', 999)))

        # Recommendation text
        recommendations = f"""
TOP RECOMMENDATIONS:

Best Risk-Adjusted Return:
   • Strategy: {best_sharpe[0]}
   • Sharpe Ratio: {best_sharpe[1]['performance_metrics'].get('sharpe_ratio', 0):.3f}
   • Net PnL: ${best_sharpe[1]['performance_metrics'].get('net_total_pnl', 0):,.0f}

Highest Net PnL:
   • Strategy: {best_pnl[0]}
   • Net PnL: ${best_pnl[1]['performance_metrics'].get('net_total_pnl', 0):,.0f}
   • Return: {best_pnl[1]['performance_metrics'].get('cumulative_return', 0) * 100:.2f}%

Lowest Risk (Min Drawdown):
   • Strategy: {best_drawdown[0]}
   • Max Drawdown: {abs(best_drawdown[1]['performance_metrics'].get('max_drawdown', 0)) * 100:.2f}%
   • Volatility: {best_drawdown[1]['performance_metrics'].get('volatility', 0) * 100:.2f}%

IMPORTANT CONSIDERATIONS:

• Transaction Costs Matter: All results shown are NET of costs
• Risk Management: Consider maximum drawdown for position sizing
• Market Conditions: Past performance may not predict future results
• Diversification: Consider combining strategies for better risk-return profile

STRATEGY SELECTION GUIDE:

• Conservative Investors: Choose lowest drawdown strategy
• Aggressive Investors: Focus on highest Sharpe ratio
• Capital Growth Focus: Select highest Net PnL strategy
• Balanced Approach: Weight multiple top-performing strategies
        """

        ax.text(0.05, 0.85, recommendations, fontsize=12, transform=ax.transAxes,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))

        # Bottom disclaimer
        disclaimer = ("This analysis is for educational purposes only. "
                     "Please perform your own due diligence before making investment decisions.")
        ax.text(0.5, 0.05, disclaimer, fontsize=10, ha='center', style='italic',
                transform=ax.transAxes, alpha=0.7)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def generate_sample_strategy_comparison_pdf():
    """Generate sample strategy comparison PDF"""
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    strategies_results = {
        'Momentum Strategy': {
            'performance_metrics': {
                'net_total_pnl': 2500,
                'cumulative_return': 0.025,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.08,
                'volatility': 0.15,
                'win_rate': 0.55,
                'cost_drag_pct': 2.1,
                'net_profit_margin': 12.5,
                'return_to_cost_ratio': 8.2
            },
            'returns_series': pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
        },
        'Mean Reversion': {
            'performance_metrics': {
                'net_total_pnl': 1800,
                'cumulative_return': 0.018,
                'sharpe_ratio': 1.45,
                'max_drawdown': -0.05,
                'volatility': 0.12,
                'win_rate': 0.58,
                'cost_drag_pct': 1.8,
                'net_profit_margin': 15.2,
                'return_to_cost_ratio': 10.1
            },
            'returns_series': pd.Series(np.random.normal(0.0008, 0.015, 100), index=dates)
        },
        'ML Ridge': {
            'performance_metrics': {
                'net_total_pnl': 3200,
                'cumulative_return': 0.032,
                'sharpe_ratio': 1.65,
                'max_drawdown': -0.06,
                'volatility': 0.18,
                'win_rate': 0.62,
                'cost_drag_pct': 2.5,
                'net_profit_margin': 18.7,
                'return_to_cost_ratio': 12.8
            },
            'returns_series': pd.Series(np.random.normal(0.0012, 0.022, 100), index=dates)
        }
    }

    generator = StrategyComparisonPDFGenerator()
    return generator.generate_strategy_comparison_pdf(
        strategies_results,
        "sample_strategy_comparison.pdf"
    )


if __name__ == "__main__":
    generate_sample_strategy_comparison_pdf()
    print("Sample strategy comparison PDF report generated successfully!")