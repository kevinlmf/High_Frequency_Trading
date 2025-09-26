"""
Net PnL PDF Report Generator for HFT Trading System
Professional Net PnL PDF report generator for comprehensive analysis
"""

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import warnings

# Set style for professional charts
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class NetPnLPDFReportGenerator:
    """
    Professional Net PnL PDF report generator
    Generate comprehensive PDF reports with all Net PnL related metrics and visualizations
    """

    def __init__(self, strategy_name: str = "HFT Strategy",
                 report_title: str = "Net PnL Performance Report"):
        """
        Initialize PDF report generator

        Args:
            strategy_name: Name of the strategy
            report_title: Title of the report
        """
        self.strategy_name = strategy_name
        self.report_title = report_title
        self.fig_size = (12, 8)
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

        # Set font support
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def generate_comprehensive_pdf_report(self,
                                        portfolio_history: pd.DataFrame,
                                        performance_metrics: Dict[str, Any],
                                        trades_data: Optional[pd.DataFrame] = None,
                                        benchmark_data: Optional[pd.Series] = None,
                                        save_path: str = "net_pnl_report.pdf") -> str:
        """
        Generate comprehensive Net PnL PDF report

        Args:
            portfolio_history: Portfolio historical data containing Net PnL columns
            performance_metrics: Performance metrics dictionary
            trades_data: Trading data (optional)
            benchmark_data: Benchmark data (optional)
            save_path: PDF save path

        Returns:
            Generated PDF file path
        """

        with PdfPages(save_path) as pdf:
            # Page 1: Cover page and executive summary
            self._create_cover_page(pdf, performance_metrics)

            # Page 2: Net PnL trend chart
            self._create_net_pnl_trend_chart(pdf, portfolio_history, benchmark_data)

            # Page 3: Risk-return analysis
            self._create_risk_return_analysis(pdf, portfolio_history, performance_metrics)

            # Page 4: Drawdown analysis
            self._create_drawdown_analysis(pdf, portfolio_history)

            # Page 5: Cost analysis
            self._create_cost_analysis(pdf, performance_metrics)

            # Page 6: Rolling metrics analysis
            self._create_rolling_metrics(pdf, portfolio_history)

            # Page 7: Trade analysis (if trade data available)
            if trades_data is not None:
                self._create_trade_analysis(pdf, trades_data)

            # Page 8: Detailed metrics table
            self._create_detailed_metrics_table(pdf, performance_metrics)

        print(f"Net PnL PDF report generated: {save_path}")
        return save_path

    def _create_cover_page(self, pdf: PdfPages, performance_metrics: Dict[str, Any]):
        """Create cover page"""
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.axis('off')

        # Title
        ax.text(0.5, 0.9, self.report_title, fontsize=24, fontweight='bold',
                ha='center', transform=ax.transAxes)
        ax.text(0.5, 0.85, f"Strategy: {self.strategy_name}", fontsize=16,
                ha='center', transform=ax.transAxes, style='italic')

        # Generation time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(0.5, 0.8, f"Generated: {current_time}", fontsize=12,
                ha='center', transform=ax.transAxes)

        # Core metrics summary
        ax.text(0.5, 0.7, "Executive Summary", fontsize=18, fontweight='bold',
                ha='center', transform=ax.transAxes)

        # Extract key metrics
        net_pnl = performance_metrics.get('net_total_pnl', 0)
        net_return = performance_metrics.get('cumulative_return', 0) * 100
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        max_drawdown = performance_metrics.get('max_drawdown', 0) * 100
        cost_drag = performance_metrics.get('cost_drag_pct', 0)

        # Key metrics display
        metrics_text = f"""
Net PnL: ${net_pnl:,.2f}
Net Return: {net_return:.2f}%
Sharpe Ratio: {sharpe_ratio:.3f}
Max Drawdown: {abs(max_drawdown):.2f}%
Cost Drag: {cost_drag:.2f}%
        """

        ax.text(0.5, 0.5, metrics_text, fontsize=14, ha='center',
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5",
                facecolor='lightblue', alpha=0.3))

        # Performance rating
        rating = self._calculate_performance_rating(sharpe_ratio, max_drawdown/100, cost_drag)
        ax.text(0.5, 0.25, f"Overall Rating: {rating}", fontsize=16, fontweight='bold',
                ha='center', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.5))

        # Disclaimer
        disclaimer = "This report is for informational purposes only. Past performance does not guarantee future results."
        ax.text(0.5, 0.05, disclaimer, fontsize=8, ha='center',
                transform=ax.transAxes, style='italic', alpha=0.7)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_net_pnl_trend_chart(self, pdf: PdfPages, portfolio_history: pd.DataFrame,
                                  benchmark_data: Optional[pd.Series] = None):
        """Create Net PnL trend chart"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.fig_size, height_ratios=[2, 1])

        # Main chart: Net PnL trend
        if 'net_total_pnl' in portfolio_history.columns:
            ax1.plot(portfolio_history.index, portfolio_history['net_total_pnl'],
                    color=self.colors[0], linewidth=2, label='Net PnL')
            ax1.fill_between(portfolio_history.index, 0, portfolio_history['net_total_pnl'],
                           alpha=0.3, color=self.colors[0])

        if 'gross_pnl' in portfolio_history.columns:
            ax1.plot(portfolio_history.index, portfolio_history['gross_pnl'],
                    color=self.colors[1], linewidth=1.5, linestyle='--', label='Gross PnL')

        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_title('Net PnL Trend Over Time', fontsize=16, fontweight='bold')
        ax1.set_ylabel('PnL ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Sub-chart: Net return percentage
        if 'net_return' in portfolio_history.columns:
            ax2.plot(portfolio_history.index, portfolio_history['net_return'],
                    color=self.colors[2], linewidth=2, label='Net Return %')
        elif 'net_total_pnl' in portfolio_history.columns:
            # Calculate return percentage if not available
            initial_capital = portfolio_history['total_value'].iloc[0] if 'total_value' in portfolio_history.columns else 100000
            net_return_pct = (portfolio_history['net_total_pnl'] / initial_capital) * 100
            ax2.plot(portfolio_history.index, net_return_pct,
                    color=self.colors[2], linewidth=2, label='Net Return %')

        if benchmark_data is not None:
            benchmark_cumret = (1 + benchmark_data).cumprod() - 1
            ax2.plot(benchmark_cumret.index, benchmark_cumret.values * 100,
                    color='gray', linewidth=1.5, linestyle=':', label='Benchmark')

        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Return (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_risk_return_analysis(self, pdf: PdfPages, portfolio_history: pd.DataFrame,
                                   performance_metrics: Dict[str, Any]):
        """Create risk-return analysis page"""
        fig = plt.figure(figsize=self.fig_size)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Return distribution histogram
        ax1 = fig.add_subplot(gs[0, 0])
        if 'net_daily_pnl' in portfolio_history.columns:
            daily_returns = portfolio_history['net_daily_pnl'].dropna()
            ax1.hist(daily_returns, bins=30, alpha=0.7, color=self.colors[0], edgecolor='black')
            ax1.axvline(daily_returns.mean(), color='red', linestyle='--',
                       label=f'Mean: ${daily_returns.mean():.2f}')
            ax1.set_title('Daily Net PnL Distribution', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Daily Net PnL ($)')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. VaR and CVaR visualization
        ax2 = fig.add_subplot(gs[0, 1])
        if 'net_daily_pnl' in portfolio_history.columns:
            daily_returns = portfolio_history['net_daily_pnl'].dropna()
            var_5 = np.percentile(daily_returns, 5)
            cvar_5 = daily_returns[daily_returns <= var_5].mean()

            ax2.hist(daily_returns, bins=30, alpha=0.7, color=self.colors[1], edgecolor='black')
            ax2.axvline(var_5, color='red', linestyle='-', linewidth=2,
                       label=f'VaR 5%: ${var_5:.2f}')
            ax2.axvline(cvar_5, color='darkred', linestyle='--', linewidth=2,
                       label=f'CVaR 5%: ${cvar_5:.2f}')

            # Fill VaR area
            x_var = daily_returns[daily_returns <= var_5]
            if len(x_var) > 0:
                ax2.hist(x_var, bins=15, alpha=0.8, color='red', edgecolor='darkred')

            ax2.set_title('VaR & CVaR Analysis', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Daily Net PnL ($)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. Rolling volatility
        ax3 = fig.add_subplot(gs[1, 0])
        if 'net_daily_pnl' in portfolio_history.columns:
            daily_returns = portfolio_history['net_daily_pnl'].dropna()
            rolling_vol = daily_returns.rolling(window=20).std()
            ax3.plot(rolling_vol.index, rolling_vol, color=self.colors[2], linewidth=2)
            ax3.fill_between(rolling_vol.index, rolling_vol, alpha=0.3, color=self.colors[2])
            ax3.set_title('20-Day Rolling Volatility', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Volatility ($)')
            ax3.grid(True, alpha=0.3)

        # 4. Key risk indicators
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        # Risk metrics text
        risk_metrics = [
            ('Sharpe Ratio', performance_metrics.get('sharpe_ratio', 0), ':.3f'),
            ('Max Drawdown', abs(performance_metrics.get('max_drawdown', 0)) * 100, ':.2f%'),
            ('Calmar Ratio', performance_metrics.get('calmar_ratio', 0), ':.3f'),
            ('Volatility', performance_metrics.get('volatility', 0) * 100, ':.2f%'),
            ('Skewness', performance_metrics.get('skewness', 0), ':.3f'),
            ('Kurtosis', performance_metrics.get('kurtosis', 0), ':.3f')
        ]

        y_pos = 0.9
        ax4.text(0.5, 0.95, 'Risk Metrics Summary', fontsize=14, fontweight='bold',
                ha='center', transform=ax4.transAxes)

        for metric_name, value, fmt in risk_metrics:
            ax4.text(0.1, y_pos, f"{metric_name}:", fontsize=11, fontweight='bold',
                    transform=ax4.transAxes)

            # Format numeric values
            if fmt == ':.3f':
                formatted_value = f"{value:.3f}"
            elif fmt == ':.2f%':
                formatted_value = f"{value:.2f}%"
            else:
                formatted_value = str(value)

            ax4.text(0.7, y_pos, formatted_value, fontsize=11,
                    transform=ax4.transAxes)
            y_pos -= 0.12

        plt.suptitle('Risk-Return Analysis', fontsize=16, fontweight='bold', y=0.98)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_drawdown_analysis(self, pdf: PdfPages, portfolio_history: pd.DataFrame):
        """Create drawdown analysis page"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.fig_size, height_ratios=[2, 1])

        # Calculate drawdown
        if 'net_total_pnl' in portfolio_history.columns:
            cumulative_pnl = portfolio_history['net_total_pnl']
            rolling_max = cumulative_pnl.expanding().max()
            drawdown = (cumulative_pnl - rolling_max)
            drawdown_pct = drawdown / rolling_max * 100

            # Main chart: Net value and drawdown
            ax1_twin = ax1.twinx()

            # Net value line
            ax1.plot(cumulative_pnl.index, cumulative_pnl, color=self.colors[0],
                    linewidth=2, label='Net PnL')
            ax1.plot(rolling_max.index, rolling_max, color=self.colors[1],
                    linewidth=1.5, linestyle='--', alpha=0.7, label='Peak')

            # Drawdown area
            ax1_twin.fill_between(drawdown_pct.index, 0, drawdown_pct,
                                 color='red', alpha=0.3, label='Drawdown %')
            ax1_twin.plot(drawdown_pct.index, drawdown_pct, color='red', linewidth=1)

            ax1.set_title('Net PnL and Drawdown Analysis', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Net PnL ($)', color=self.colors[0])
            ax1_twin.set_ylabel('Drawdown (%)', color='red')
            ax1.grid(True, alpha=0.3)

            # Legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            # Sub-chart: Drawdown duration analysis
            drawdown_periods = self._analyze_drawdown_periods(drawdown_pct)
            if drawdown_periods:
                durations = [period['duration'] for period in drawdown_periods]
                depths = [abs(period['max_depth']) for period in drawdown_periods]

                scatter = ax2.scatter(durations, depths, c=depths, cmap='Reds',
                                    s=100, alpha=0.7, edgecolors='black')
                ax2.set_xlabel('Drawdown Duration (Days)', fontsize=12)
                ax2.set_ylabel('Max Drawdown Depth (%)', fontsize=12)
                ax2.set_title('Drawdown Duration vs Depth', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)

                # Add color bar
                plt.colorbar(scatter, ax=ax2, label='Depth (%)')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_cost_analysis(self, pdf: PdfPages, performance_metrics: Dict[str, Any]):
        """Create cost analysis page"""
        fig = plt.figure(figsize=self.fig_size)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Cost structure pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        cost_components = {
            'Commission': performance_metrics.get('total_commission', 0),
            'Slippage': performance_metrics.get('total_slippage', 0),
            'Other Costs': performance_metrics.get('other_costs', 0)
        }

        # Filter out zero costs
        cost_components = {k: v for k, v in cost_components.items() if v > 0}

        if cost_components:
            wedges, texts, autotexts = ax1.pie(cost_components.values(),
                                             labels=cost_components.keys(),
                                             autopct='%1.1f%%',
                                             colors=self.colors[:len(cost_components)])
            ax1.set_title('Cost Structure Breakdown', fontsize=12, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No detailed cost\nbreakdown available',
                    ha='center', va='center', fontsize=12,
                    transform=ax1.transAxes)
            ax1.set_title('Cost Structure Breakdown', fontsize=12, fontweight='bold')

        # 2. Gross profit vs net profit comparison
        ax2 = fig.add_subplot(gs[0, 1])
        gross_return = performance_metrics.get('gross_cumulative_return', 0) * 100
        net_return = performance_metrics.get('cumulative_return', 0) * 100
        cost_drag = performance_metrics.get('cost_drag_pct', 0)

        categories = ['Gross Return', 'Net Return']
        values = [gross_return, net_return]
        colors = [self.colors[0], self.colors[1]]

        bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')

        ax2.set_title('Gross vs Net Return', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add cost drag annotation
        if cost_drag > 0:
            ax2.text(0.5, 0.8, f'Cost Drag: {cost_drag:.2f}%',
                    transform=ax2.transAxes, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))

        # 3. Cost efficiency metrics
        ax3 = fig.add_subplot(gs[1, :])

        cost_metrics = [
            ('Cost to Capital Ratio', performance_metrics.get('cost_to_capital_ratio', 0) * 100, '%'),
            ('Return to Cost Ratio', performance_metrics.get('return_to_cost_ratio', 0), 'x'),
            ('Net Profit Margin', performance_metrics.get('net_profit_margin', 0), '%'),
            ('Cost Drag Percentage', performance_metrics.get('cost_drag_pct', 0), '%'),
            ('Breakeven Trades', performance_metrics.get('breakeven_trades', 0), 'trades')
        ]

        # Create table
        table_data = []
        for metric_name, value, unit in cost_metrics:
            if unit == '%':
                formatted_value = f"{value:.2f}%"
            elif unit == 'x':
                formatted_value = f"{value:.2f}x"
            elif unit == 'trades':
                if value == float('inf'):
                    formatted_value = "Cannot break even"
                else:
                    formatted_value = f"{int(value)} trades"
            else:
                formatted_value = f"{value:.4f}"

            table_data.append([metric_name, formatted_value])

        # Create table
        table = ax3.table(cellText=table_data,
                         colLabels=['Cost Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.6, 0.4])

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)

        # Set table style
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#E6E6FA')
            else:
                cell.set_facecolor('#F8F8FF')
            cell.set_edgecolor('#CCCCCC')

        ax3.axis('off')
        ax3.set_title('Cost Efficiency Metrics', fontsize=14, fontweight='bold', pad=20)

        plt.suptitle('Cost Analysis', fontsize=16, fontweight='bold', y=0.98)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_rolling_metrics(self, pdf: PdfPages, portfolio_history: pd.DataFrame, window: int = 20):
        """Create rolling metrics analysis page"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.fig_size)

        if 'net_daily_pnl' in portfolio_history.columns:
            daily_pnl = portfolio_history['net_daily_pnl'].dropna()

            # 1. Rolling Sharpe ratio
            rolling_mean = daily_pnl.rolling(window).mean()
            rolling_std = daily_pnl.rolling(window).std()
            rolling_sharpe = (rolling_mean * 252 - 0.02) / (rolling_std * np.sqrt(252))

            ax1.plot(rolling_sharpe.index, rolling_sharpe, color=self.colors[0], linewidth=2)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax1.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Good (>1.0)')
            ax1.axhline(y=2, color='blue', linestyle='--', alpha=0.7, label='Excellent (>2.0)')
            ax1.fill_between(rolling_sharpe.index, 0, rolling_sharpe,
                           where=(rolling_sharpe > 0), alpha=0.3, color='green')
            ax1.fill_between(rolling_sharpe.index, 0, rolling_sharpe,
                           where=(rolling_sharpe < 0), alpha=0.3, color='red')
            ax1.set_title(f'{window}-Day Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Sharpe Ratio')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Rolling volatility
            rolling_vol = daily_pnl.rolling(window).std() * np.sqrt(252)
            ax2.plot(rolling_vol.index, rolling_vol, color=self.colors[1], linewidth=2)
            ax2.fill_between(rolling_vol.index, rolling_vol, alpha=0.3, color=self.colors[1])
            ax2.set_title(f'{window}-Day Rolling Volatility', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Annualized Volatility')
            ax2.grid(True, alpha=0.3)

            # 3. Rolling win rate
            rolling_win_rate = (daily_pnl > 0).rolling(window).mean() * 100
            ax3.plot(rolling_win_rate.index, rolling_win_rate, color=self.colors[2], linewidth=2)
            ax3.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Break-even (50%)')
            ax3.fill_between(rolling_win_rate.index, 50, rolling_win_rate,
                           where=(rolling_win_rate > 50), alpha=0.3, color='green')
            ax3.fill_between(rolling_win_rate.index, 50, rolling_win_rate,
                           where=(rolling_win_rate < 50), alpha=0.3, color='red')
            ax3.set_title(f'{window}-Day Rolling Win Rate', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Win Rate (%)')
            ax3.set_xlabel('Date')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 4. Rolling return/risk ratio
            rolling_return_risk = rolling_mean / rolling_std
            ax4.plot(rolling_return_risk.index, rolling_return_risk, color=self.colors[3], linewidth=2)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.fill_between(rolling_return_risk.index, 0, rolling_return_risk,
                           where=(rolling_return_risk > 0), alpha=0.3, color='green')
            ax4.fill_between(rolling_return_risk.index, 0, rolling_return_risk,
                           where=(rolling_return_risk < 0), alpha=0.3, color='red')
            ax4.set_title(f'{window}-Day Return/Risk Ratio', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Return/Risk Ratio')
            ax4.set_xlabel('Date')
            ax4.grid(True, alpha=0.3)

        plt.suptitle(f'Rolling Performance Metrics (Window: {window} days)',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_trade_analysis(self, pdf: PdfPages, trades_data: pd.DataFrame):
        """Create trade analysis page"""
        fig = plt.figure(figsize=self.fig_size)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Trade volume distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if 'quantity' in trades_data.columns and 'execution_price' in trades_data.columns:
            trade_values = trades_data['quantity'] * trades_data['execution_price']
            ax1.hist(trade_values, bins=30, alpha=0.7, color=self.colors[0], edgecolor='black')
            ax1.set_title('Trade Size Distribution', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Trade Value ($)')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)

        # 2. Trading time analysis
        ax2 = fig.add_subplot(gs[0, 1])
        if 'timestamp' in trades_data.columns:
            trades_copy = trades_data.copy()
            trades_copy['timestamp'] = pd.to_datetime(trades_copy['timestamp'])
            trades_copy['hour'] = trades_copy['timestamp'].dt.hour
            hourly_trades = trades_copy['hour'].value_counts().sort_index()

            bars = ax2.bar(hourly_trades.index, hourly_trades.values,
                          color=self.colors[1], alpha=0.7, edgecolor='black')
            ax2.set_title('Trading Activity by Hour', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Number of Trades')
            ax2.grid(True, alpha=0.3, axis='y')

        # 3. Slippage analysis
        ax3 = fig.add_subplot(gs[1, 0])
        if 'slippage' in trades_data.columns:
            slippage_data = trades_data['slippage'].dropna()
            ax3.hist(slippage_data, bins=30, alpha=0.7, color=self.colors[2], edgecolor='black')
            ax3.axvline(slippage_data.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: ${slippage_data.mean():.4f}')
            ax3.set_title('Slippage Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Slippage ($)')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Commission analysis
        ax4 = fig.add_subplot(gs[1, 1])
        if 'commission' in trades_data.columns:
            commission_data = trades_data['commission'].dropna()
            ax4.hist(commission_data, bins=30, alpha=0.7, color=self.colors[3], edgecolor='black')
            ax4.axvline(commission_data.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: ${commission_data.mean():.4f}')
            ax4.set_title('Commission Distribution', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Commission ($)')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.suptitle('Trading Activity Analysis', fontsize=16, fontweight='bold', y=0.98)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_detailed_metrics_table(self, pdf: PdfPages, performance_metrics: Dict[str, Any]):
        """Create detailed metrics table page"""
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.axis('off')

        # Categorize metrics
        metrics_categories = {
            'Return Metrics': {
                'Net Total PnL': ('$', performance_metrics.get('net_total_pnl', 0)),
                'Cumulative Return': ('%', performance_metrics.get('cumulative_return', 0) * 100),
                'Annualized Return': ('%', performance_metrics.get('annualized_return', 0) * 100),
                'Average Daily Return': ('%', performance_metrics.get('average_daily_return', 0) * 100)
            },
            'Risk Metrics': {
                'Volatility': ('%', performance_metrics.get('volatility', 0) * 100),
                'Sharpe Ratio': ('', performance_metrics.get('sharpe_ratio', 0)),
                'Maximum Drawdown': ('%', abs(performance_metrics.get('max_drawdown', 0)) * 100),
                'Calmar Ratio': ('', performance_metrics.get('calmar_ratio', 0)),
                'VaR 5%': ('%', performance_metrics.get('var_5pct', 0) * 100),
                'CVaR 5%': ('%', performance_metrics.get('cvar_5pct', 0) * 100)
            },
            'Trading Metrics': {
                'Win Rate': ('%', performance_metrics.get('win_rate', 0) * 100),
                'Profit/Loss Ratio': ('', performance_metrics.get('profit_loss_ratio', 0)),
                'Max Consecutive Wins': ('', performance_metrics.get('max_consecutive_wins', 0)),
                'Max Consecutive Losses': ('', performance_metrics.get('max_consecutive_losses', 0))
            },
            'Cost Metrics': {
                'Cost to Capital Ratio': ('%', performance_metrics.get('cost_to_capital_ratio', 0) * 100),
                'Cost Drag': ('%', performance_metrics.get('cost_drag_pct', 0)),
                'Net Profit Margin': ('%', performance_metrics.get('net_profit_margin', 0)),
                'Return to Cost Ratio': ('x', performance_metrics.get('return_to_cost_ratio', 0))
            }
        }

        y_start = 0.95
        y_current = y_start

        ax.text(0.5, 0.98, 'Detailed Performance Metrics', fontsize=18, fontweight='bold',
                ha='center', transform=ax.transAxes)

        for category, metrics in metrics_categories.items():
            # Category title
            ax.text(0.05, y_current, category, fontsize=14, fontweight='bold',
                   transform=ax.transAxes, color=self.colors[0])
            y_current -= 0.03

            # Draw separator line
            ax.plot([0.05, 0.95], [y_current, y_current], color='gray',
                   linewidth=1, alpha=0.5, transform=ax.transAxes)
            y_current -= 0.02

            # Metrics list
            for metric_name, (unit, value) in metrics.items():
                # Format numeric values
                if unit == '$':
                    formatted_value = f"${value:,.2f}"
                elif unit == '%':
                    formatted_value = f"{value:.2f}%"
                elif unit == 'x':
                    formatted_value = f"{value:.2f}x"
                else:
                    formatted_value = f"{value:.4f}"

                ax.text(0.1, y_current, f"{metric_name}:", fontsize=11,
                       transform=ax.transAxes)
                ax.text(0.7, y_current, formatted_value, fontsize=11,
                       transform=ax.transAxes, fontweight='bold')
                y_current -= 0.025

            y_current -= 0.02  # Category spacing

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _calculate_performance_rating(self, sharpe_ratio: float, max_drawdown: float, cost_drag: float) -> str:
        """Calculate performance rating"""
        score = 0

        # Sharpe ratio scoring (0-40 points)
        if sharpe_ratio >= 2.0:
            score += 40
        elif sharpe_ratio >= 1.5:
            score += 30
        elif sharpe_ratio >= 1.0:
            score += 20
        elif sharpe_ratio >= 0.5:
            score += 10

        # Maximum drawdown scoring (0-35 points)
        if abs(max_drawdown) <= 0.05:
            score += 35
        elif abs(max_drawdown) <= 0.10:
            score += 25
        elif abs(max_drawdown) <= 0.20:
            score += 15
        elif abs(max_drawdown) <= 0.30:
            score += 5

        # Cost drag scoring (0-25 points)
        if cost_drag <= 1.0:
            score += 25
        elif cost_drag <= 3.0:
            score += 20
        elif cost_drag <= 5.0:
            score += 10
        elif cost_drag <= 10.0:
            score += 5

        # Rating mapping
        if score >= 85:
            return "Excellent (A+)"
        elif score >= 70:
            return "Very Good (A)"
        elif score >= 55:
            return "Good (B)"
        elif score >= 40:
            return "Fair (C)"
        else:
            return "Poor (D)"

    def _analyze_drawdown_periods(self, drawdown_series: pd.Series) -> List[Dict]:
        """Analyze drawdown periods"""
        periods = []
        in_drawdown = False
        start_date = None
        max_depth = 0

        for date, dd in drawdown_series.items():
            if dd < -0.1 and not in_drawdown:  # Start drawdown (>0.1%)
                in_drawdown = True
                start_date = date
                max_depth = dd
            elif in_drawdown:
                max_depth = min(max_depth, dd)
                if dd >= -0.01:  # End drawdown (<0.01%)
                    periods.append({
                        'start': start_date,
                        'end': date,
                        'duration': (date - start_date).days,
                        'max_depth': max_depth
                    })
                    in_drawdown = False
                    max_depth = 0

        return periods

# Example usage function
def generate_sample_pdf_report():
    """Generate sample PDF report"""

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=250, freq='D')
    np.random.seed(42)

    # Simulate portfolio historical data
    returns = np.random.normal(0.001, 0.02, 250)
    cumulative_pnl = np.cumsum(returns * 100000)  # Assume initial capital 100k

    portfolio_history = pd.DataFrame({
        'net_total_pnl': cumulative_pnl,
        'net_daily_pnl': returns * 100000,
        'total_value': 100000 + cumulative_pnl,
        'net_return': (cumulative_pnl / 100000) * 100
    }, index=dates)

    # Simulate performance metrics
    performance_metrics = {
        'net_total_pnl': cumulative_pnl[-1],
        'cumulative_return': (cumulative_pnl[-1] / 100000),
        'annualized_return': 0.12,
        'sharpe_ratio': 1.5,
        'max_drawdown': -0.08,
        'volatility': 0.15,
        'var_5pct': -0.025,
        'cvar_5pct': -0.035,
        'win_rate': 0.55,
        'profit_loss_ratio': 1.3,
        'cost_to_capital_ratio': 0.002,
        'cost_drag_pct': 2.5,
        'net_profit_margin': 8.5,
        'return_to_cost_ratio': 15.2
    }

    # Generate report
    generator = NetPnLPDFReportGenerator("Sample HFT Strategy", "Net PnL Performance Report")
    report_path = generator.generate_comprehensive_pdf_report(
        portfolio_history=portfolio_history,
        performance_metrics=performance_metrics,
        save_path="sample_net_pnl_report.pdf"
    )

    return report_path

if __name__ == "__main__":
    generate_sample_pdf_report()
    print("Sample PDF report generation completed!")