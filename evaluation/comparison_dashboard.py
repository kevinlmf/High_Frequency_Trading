"""
Strategy Comparison Dashboard for HFT Trading System
strategyAnalyze
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
from .performance_metrics import PerformanceMetrics, calculate_strategy_comparison

# SetupSupports
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class StrategyComparisonDashboard:
    """
strategy
"""

    def __init__(self, strategies_data: Dict[str, Dict[str, Any]]):
        """
Initialize

        Args:
            strategies_data: {
                'strategy_name': {
                    'returns': pd.Series,
                    'portfolio_history': pd.DataFrame,
                    'trades': pd.DataFrame,
                    'performance_metrics': Dict,
                    'positions': Dict,
                    ...
                }
            }
"""
        self.strategies_data = strategies_data
        self.strategy_names = list(strategies_data.keys())

        # Validatedata
        self._validate_data()

    def _validate_data(self):
        """
Validatedata
"""
        required_fields = ['returns', 'performance_metrics']

        for strategy_name, data in self.strategies_data.items():
            for field in required_fields:
                if field not in data:
                    warnings.warn(f"strategy {strategy_name} : {field}")

    def generate_performance_comparison_table(self) -> pd.DataFrame:
        """
Generate
"""
        comparison_data = {}

        for strategy_name, data in self.strategies_data.items():
            metrics = data.get('performance_metrics', {})
            comparison_data[strategy_name] = metrics

        comparison_df = pd.DataFrame(comparison_data).T

        # metrics
        key_metrics = [
            'cumulative_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'max_drawdown', 'calmar_ratio',
            'win_rate', 'profit_loss_ratio', 'var_5pct', 'cvar_5pct'
        ]

        formatted_comparison = pd.DataFrame(index=comparison_df.index)

        for metric in key_metrics:
            if metric in comparison_df.columns:
                if metric in ['cumulative_return', 'annualized_return', 'volatility',
                             'max_drawdown', 'win_rate', 'var_5pct', 'cvar_5pct']:
                    # 
                    formatted_comparison[metric] = comparison_df[metric].apply(lambda x: f"{x:.2%}")
                else:
                    # 
                    formatted_comparison[metric] = comparison_df[metric].apply(lambda x: f"{x:.3f}")

        return formatted_comparison

    def plot_cumulative_returns(self, figsize: Tuple[int, int] = (12, 8),
                               save_path: Optional[str] = None) -> go.Figure:
        """

"""
        fig = go.Figure()

        for strategy_name, data in self.strategies_data.items():
            returns = data.get('returns')
            if returns is not None and not returns.empty:
                cum_returns = (1 + returns).cumprod() - 1
                fig.add_trace(go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns.values * 100,
                    name=strategy_name,
                    mode='lines',
                    line=dict(width=2)
                ))

        fig.update_layout(
            title='strategy',
            xaxis_title='Date',
            yaxis_title=' (%)',
            width=figsize[0] * 80,
            height=figsize[1] * 80,
            template='plotly_white',
            legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)')
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_drawdown_comparison(self, figsize: Tuple[int, int] = (12, 6),
                               save_path: Optional[str] = None) -> go.Figure:
        """

"""
        fig = go.Figure()

        for strategy_name, data in self.strategies_data.items():
            returns = data.get('returns')
            if returns is not None and not returns.empty:
                cum_returns = (1 + returns).cumprod()
                rolling_max = cum_returns.expanding().max()
                drawdown = (cum_returns - rolling_max) / rolling_max * 100

                fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    name=strategy_name,
                    mode='lines',
                    line=dict(width=2),
                    fill='tonexty' if strategy_name == list(self.strategies_data.keys())[0] else None
                ))

        fig.update_layout(
            title='strategy',
            xaxis_title='Date',
            yaxis_title=' (%)',
            width=figsize[0] * 80,
            height=figsize[1] * 80,
            template='plotly_white',
            legend=dict(x=0, y=-0.3, bgcolor='rgba(255,255,255,0.8)')
        )

        # 0
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_risk_return_scatter(self, figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[str] = None) -> go.Figure:
        """

"""
        risk_return_data = []

        for strategy_name, data in self.strategies_data.items():
            metrics = data.get('performance_metrics', {})
            risk_return_data.append({
                'strategy': strategy_name,
                'return': metrics.get('annualized_return', 0) * 100,
                'risk': metrics.get('volatility', 0) * 100,
                'sharpe': metrics.get('sharpe_ratio', 0),
                'max_drawdown': abs(metrics.get('max_drawdown', 0)) * 100
            })

        df = pd.DataFrame(risk_return_data)

        fig = go.Figure()

        # 
        fig.add_trace(go.Scatter(
            x=df['risk'],
            y=df['return'],
            mode='markers+text',
            marker=dict(
                size=df['sharpe'] * 20 + 20,  # Sharpe ratio
                color=df['max_drawdown'],  # Maximum drawdown
                colorscale='Viridis_r',
                showscale=True,
                colorbar=dict(title="Maximum drawdown (%)"),
                line=dict(width=2, color='white')
            ),
            text=df['strategy'],
            textposition='top center',
            name='strategy',
            hovertemplate=(
                '<b>%{text}</b><br>' +
                'Annualized return: %{y:.2f}%<br>' +
                ': %{x:.2f}%<br>' +
                'Sharpe ratio: %{customdata[0]:.3f}<br>' +
                'Maximum drawdown: %{customdata[1]:.2f}%<br>' +
                '<extra></extra>'
            ),
            customdata=df[['sharpe', 'max_drawdown']].values
        ))

        fig.update_layout(
            title='strategy',
            xaxis_title=' (%)',
            yaxis_title='Annualized return (%)',
            width=figsize[0] * 80,
            height=figsize[1] * 80,
            template='plotly_white'
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_rolling_sharpe(self, window: int = 30, figsize: Tuple[int, int] = (12, 6),
                          save_path: Optional[str] = None) -> go.Figure:
        """
Sharpe ratio
"""
        fig = go.Figure()

        for strategy_name, data in self.strategies_data.items():
            returns = data.get('returns')
            if returns is not None and not returns.empty:
                rolling_returns = returns.rolling(window)
                rolling_sharpe = (rolling_returns.mean() * 252 - 0.02) / (rolling_returns.std() * np.sqrt(252))

                fig.add_trace(go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe.values,
                    name=strategy_name,
                    mode='lines',
                    line=dict(width=2)
                ))

        fig.update_layout(
            title=f'Sharpe ratio (: {window})',
            xaxis_title='Date',
            yaxis_title='Sharpe ratio',
            width=figsize[0] * 80,
            height=figsize[1] * 80,
            template='plotly_white',
            legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)')
        )

        # 
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=1, line_dash="dash", line_color="green", opacity=0.5)
        fig.add_hline(y=2, line_dash="dash", line_color="blue", opacity=0.5)

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_monthly_returns_heatmap(self, strategy_name: str,
                                   figsize: Tuple[int, int] = (10, 6),
                                   save_path: Optional[str] = None) -> go.Figure:
        """

"""
        if strategy_name not in self.strategies_data:
            raise ValueError(f"strategy {strategy_name} In")

        returns = self.strategies_data[strategy_name].get('returns')
        if returns is None or returns.empty:
            raise ValueError(f"strategy {strategy_name} Hasdata")

        # Calculate
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

        # Create-
        monthly_data = monthly_returns.to_frame('returns')
        monthly_data['year'] = monthly_data.index.year
        monthly_data['month'] = monthly_data.index.month

        pivot_data = monthly_data.pivot(index='year', columns='month', values='returns')
        pivot_data = pivot_data * 100  # 

        # 
        month_labels = ['1', '2', '3', '4', '5', '6',
                       '7', '8', '9', '10', '11', '12']

        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=month_labels,
            y=pivot_data.index,
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(title=" (%)"),
            text=pivot_data.round(2).values,
            texttemplate="%{text}%",
            textfont={"size": 10}
        ))

        fig.update_layout(
            title=f'{strategy_name} ',
            xaxis_title='',
            yaxis_title='',
            width=figsize[0] * 80,
            height=figsize[1] * 80,
            template='plotly_white'
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_trade_analysis(self, strategy_name: str,
                          figsize: Tuple[int, int] = (15, 10),
                          save_path: Optional[str] = None) -> go.Figure:
        """
Analyze
"""
        if strategy_name not in self.strategies_data:
            raise ValueError(f"strategy {strategy_name} In")

        trades_data = self.strategies_data[strategy_name].get('trades')
        if trades_data is None or trades_data.empty:
            raise ValueError(f"strategy {strategy_name} Hasdata")

        # Create
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('', 'Analyze', 'Analyze', 'Time'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "histogram"}]]
        )

        # 
        trade_values = trades_data['quantity'] * trades_data['execution_price']
        fig.add_trace(
            go.Histogram(x=trade_values, name='', nbinsx=30),
            row=1, col=1
        )

        # Analyze
        if 'slippage' in trades_data.columns:
            fig.add_trace(
                go.Histogram(x=trades_data['slippage'], name='', nbinsx=30),
                row=1, col=2
            )

        # Analyze
        if 'commission' in trades_data.columns:
            fig.add_trace(
                go.Histogram(x=trades_data['commission'], name='', nbinsx=30),
                row=2, col=1
            )

        # Time
        if 'timestamp' in trades_data.columns:
            trades_data['hour'] = pd.to_datetime(trades_data['timestamp']).dt.hour
            hourly_counts = trades_data['hour'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(x=hourly_counts.index, y=hourly_counts.values, name=''),
                row=2, col=2
            )

        fig.update_layout(
            title=f'{strategy_name} Analyze',
            width=figsize[0] * 80,
            height=figsize[1] * 80,
            template='plotly_white'
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def generate_comprehensive_report(self, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
Generate
"""
        report = {
            'summary': self._generate_executive_summary(),
            'performance_comparison': self.generate_performance_comparison_table(),
            'risk_analysis': self._generate_risk_analysis(),
            'trade_analysis': self._generate_trade_analysis(),
            'recommendations': self._generate_recommendations()
        }

        if save_dir:
            # Save
            self.plot_cumulative_returns(save_path=f"{save_dir}/cumulative_returns.html")
            self.plot_drawdown_comparison(save_path=f"{save_dir}/drawdown_comparison.html")
            self.plot_risk_return_scatter(save_path=f"{save_dir}/risk_return_scatter.html")
            self.plot_rolling_sharpe(save_path=f"{save_dir}/rolling_sharpe.html")

            # strategyGenerateAnalyze
            for strategy_name in self.strategy_names:
                try:
                    self.plot_monthly_returns_heatmap(
                        strategy_name,
                        save_path=f"{save_dir}/{strategy_name}_monthly_heatmap.html"
                    )
                    self.plot_trade_analysis(
                        strategy_name,
                        save_path=f"{save_dir}/{strategy_name}_trade_analysis.html"
                    )
                except Exception as e:
                    warnings.warn(f"Generatestrategy {strategy_name} AnalyzeFailed: {e}")

        return report

    def _generate_executive_summary(self) -> str:
        """
Generate
"""
        summary = "=== strategy ===\n\n"

        # strategy
        best_sharpe = -np.inf
        best_return = -np.inf
        best_sharpe_strategy = ""
        best_return_strategy = ""

        for strategy_name, data in self.strategies_data.items():
            metrics = data.get('performance_metrics', {})
            sharpe = metrics.get('sharpe_ratio', 0)
            annual_return = metrics.get('annualized_return', 0)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_sharpe_strategy = strategy_name

            if annual_return > best_return:
                best_return = annual_return
                best_return_strategy = strategy_name

        summary += f"🏆 Sharpe ratiostrategy: {best_sharpe_strategy} (Sharpe ratio: {best_sharpe:.3f})\n"
        summary += f"📈 strategy: {best_return_strategy} (Annualized return: {best_return:.2%})\n\n"

        # strategy
        summary += f"📊 strategy: {len(self.strategy_names)}\n"
        summary += f"📅 Analyze: {self._get_analysis_period()}\n\n"

        return summary

    def _generate_risk_analysis(self) -> str:
        """
GenerateAnalyze
"""
        analysis = "=== Analyze ===\n\n"

        risk_metrics = ['volatility', 'max_drawdown', 'var_5pct', 'cvar_5pct']

        for metric in risk_metrics:
            values = []
            for strategy_name, data in self.strategies_data.items():
                metrics = data.get('performance_metrics', {})
                if metric in metrics:
                    values.append((strategy_name, metrics[metric]))

            if values:
                values.sort(key=lambda x: x[1])
                best_strategy = values[0][0]
                worst_strategy = values[-1][0]

                analysis += f"{metric}:\n"
                analysis += f"  : {best_strategy} ({values[0][1]:.3%})\n"
                analysis += f"  : {worst_strategy} ({values[-1][1]:.3%})\n\n"

        return analysis

    def _generate_trade_analysis(self) -> str:
        """
GenerateAnalyze
"""
        analysis = "=== Analyze ===\n\n"

        for strategy_name, data in self.strategies_data.items():
            trades_data = data.get('trades')
            if trades_data is not None and not trades_data.empty:
                total_trades = len(trades_data)
                total_volume = (trades_data['quantity'] * trades_data['execution_price']).sum()

                analysis += f"{strategy_name}:\n"
                analysis += f"  : {total_trades}\n"
                analysis += f"  : ${total_volume:,.0f}\n"

                if 'commission' in trades_data.columns:
                    total_commission = trades_data['commission'].sum()
                    analysis += f"  : ${total_commission:,.0f}\n"

                if 'slippage' in trades_data.columns:
                    total_slippage = trades_data['slippage'].sum()
                    analysis += f"  : ${total_slippage:,.0f}\n"

                analysis += "\n"

        return analysis

    def _generate_recommendations(self) -> str:
        """
Generate
"""
        recommendations = "=== strategy ===\n\n"

        # Based onmetrics
        best_strategies = self._identify_best_strategies()

        recommendations += f"💡 strategy:\n"
        for i, (strategy, reason) in enumerate(best_strategies[:3], 1):
            recommendations += f"  {i}. {strategy}: {reason}\n"

        recommendations += "\n"

        # 
        recommendations += "⚠️ :\n"
        recommendations += "  - strategystrategy\n"
        recommendations += "  - Maximum drawdown，Setup\n"
        recommendations += "  - ，Adjuststrategyparameters\n"

        return recommendations

    def _identify_best_strategies(self) -> List[Tuple[str, str]]:
        """
strategy
"""
        strategy_scores = []

        for strategy_name, data in self.strategies_data.items():
            metrics = data.get('performance_metrics', {})

            #  (According toRequiresAdjust)
            score = (
                metrics.get('sharpe_ratio', 0) * 0.4 +
                metrics.get('annualized_return', 0) * 100 * 0.3 +
                (1 - abs(metrics.get('max_drawdown', 0))) * 0.2 +
                metrics.get('calmar_ratio', 0) * 0.1
            )

            reason = []
            if metrics.get('sharpe_ratio', 0) > 1.5:
                reason.append("Sharpe ratio")
            if metrics.get('annualized_return', 0) > 0.15:
                reason.append("Annualized return")
            if abs(metrics.get('max_drawdown', 0)) < 0.1:
                reason.append("")

            strategy_scores.append((
                strategy_name,
                ", ".join(reason) if reason else "",
                score
            ))

        # 
        strategy_scores.sort(key=lambda x: x[2], reverse=True)

        return [(name, reason) for name, reason, score in strategy_scores]

    def _get_analysis_period(self) -> str:
        """
GetAnalyze
"""
        start_dates = []
        end_dates = []

        for strategy_name, data in self.strategies_data.items():
            returns = data.get('returns')
            if returns is not None and not returns.empty:
                start_dates.append(returns.index.min())
                end_dates.append(returns.index.max())

        if start_dates and end_dates:
            overall_start = min(start_dates)
            overall_end = max(end_dates)

            # Check if dates are datetime objects, if not convert them
            if hasattr(overall_start, 'strftime'):
                return f"{overall_start.strftime('%Y-%m-%d')}  {overall_end.strftime('%Y-%m-%d')}"
            else:
                # Handle case where index is integer (step numbers)
                return f" {overall_start}  {overall_end}"
        else:
            return "data"

# ExampleUsing
if __name__ == "__main__":
    # GenerateExampledata
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')

    strategies_data = {}

    # strategy1: Momentum strategy
    momentum_returns = pd.Series(np.random.normal(0.001, 0.018, 252), index=dates)
    strategies_data['Momentum strategy'] = {
        'returns': momentum_returns,
        'performance_metrics': PerformanceMetrics(momentum_returns).calculate_all_metrics(),
        'trades': pd.DataFrame({
            'timestamp': dates[:100],
            'symbol': ['AAPL'] * 100,
            'side': ['buy', 'sell'] * 50,
            'quantity': np.random.uniform(100, 1000, 100),
            'execution_price': np.random.uniform(150, 200, 100),
            'commission': np.random.uniform(1, 10, 100),
            'slippage': np.random.uniform(0, 1, 100)
        })
    }

    # strategy2: Mean reversionstrategy
    mean_revert_returns = pd.Series(np.random.normal(0.0005, 0.015, 252), index=dates)
    strategies_data['Mean reversionstrategy'] = {
        'returns': mean_revert_returns,
        'performance_metrics': PerformanceMetrics(mean_revert_returns).calculate_all_metrics(),
        'trades': pd.DataFrame({
            'timestamp': dates[:80],
            'symbol': ['GOOGL'] * 80,
            'side': ['buy', 'sell'] * 40,
            'quantity': np.random.uniform(50, 500, 80),
            'execution_price': np.random.uniform(2000, 2500, 80),
            'commission': np.random.uniform(2, 15, 80),
            'slippage': np.random.uniform(0, 2, 80)
        })
    }

    # Create
    dashboard = StrategyComparisonDashboard(strategies_data)

    # Generate
    comparison_table = dashboard.generate_performance_comparison_table()
    print(":")
    print(comparison_table)

    # Generate
    fig1 = dashboard.plot_cumulative_returns()
    fig2 = dashboard.plot_risk_return_scatter()
    fig3 = dashboard.plot_rolling_sharpe()

    #  (RequiresInSupportsplotly)
    # fig1.show()
    # fig2.show()
    # fig3.show()

    # Generate
    report = dashboard.generate_comprehensive_report()
    print("\n" + report['summary'])
    print(report['risk_analysis'])
    print(report['recommendations'])