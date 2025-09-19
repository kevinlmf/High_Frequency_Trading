# HFT Strategy Methods

This document provides a detailed overview of the methods supported in
**HFT_Unified_System**, including their mathematical formulations and
why they are effective in the context of high-frequency trading (HFT).

------------------------------------------------------------------------

## 1. Traditional Quantitative Strategies

### 1.1 Momentum Strategy

-   **Formula:**\
    $$ r_t = \frac{P_t - P_{t-k}}{P_{t-k}} $$\
    Trade rule:
    -   Buy if $r_t > \theta$\
    -   Sell if $r_t < -\theta$
-   **Rationale in HFT:**\
    Momentum captures short-term persistence in price movements, often
    driven by order flow imbalances and high-frequency trading pressure.

------------------------------------------------------------------------

### 1.2 Mean Reversion

-   **Formula:**\
    Compute z-score deviation:\
    $$ z_t = \frac{P_t - \mu_t}{\sigma_t} $$\
    Trade rule:
    -   Enter when $|z_t| > z^*$\
    -   Close when $|z_t| \leq z^*$
-   **Rationale in HFT:**\
    Price deviations from short-term equilibrium are often corrected
    quickly. Mean reversion exploits liquidity shocks or overreactions.

------------------------------------------------------------------------

### 1.3 Pairs Trading (Statistical Arbitrage)

-   **Formula:**\
    For two cointegrated assets $A, B$:\
    $$ S_t = P^A_t - \beta P^B_t $$\
    Trade when spread $S_t$ deviates from mean:\
    $$ z_t = \frac{S_t - \mu_S}{\sigma_S} $$

-   **Rationale in HFT:**\
    Pairs trading is effective when assets have strong economic or
    statistical relationships. Deviations are exploited with fast
    reversion trades.

------------------------------------------------------------------------

## 2. Machine Learning Models

### 2.1 Linear / Ridge Regression

-   **Formula:**\
    $$ y_t = X_t \beta + \epsilon_t $$\
    Ridge adds L2 penalty:\
    $$ \min_\beta \; ||y - X\beta||^2 + \lambda ||\beta||^2 $$

-   **Rationale in HFT:**\
    Linear and ridge models capture predictive relationships between
    technical indicators and future returns, while ridge regularization
    avoids overfitting.

------------------------------------------------------------------------

### 2.2 Random Forest

-   **Formula:**\
    Ensemble of decision trees:\
    $$ \hat{y} = \frac{1}{M} \sum_{m=1}^M T_m(X) $$

-   **Rationale in HFT:**\
    Captures nonlinear interactions between features. Effective for
    noisy financial data but may require large samples.

------------------------------------------------------------------------

### 2.3 LSTM / GRU Predictors

-   **Formula:**\
    GRU hidden state update:\
    $$ h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t $$\
    where $z_t$ is the update gate.

-   **Rationale in HFT:**\
    Recurrent architectures capture sequential dependencies in
    high-frequency order flow and microstructure patterns.

------------------------------------------------------------------------

### 2.4 XGBoost

-   **Formula:**\
    Gradient boosting objective:\
    $$ \text{Obj} = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k) $$

-   **Rationale in HFT:**\
    Robust to nonlinearities, feature interactions, and can handle
    sparse data. Effective for short-term prediction tasks.

------------------------------------------------------------------------

## 3. Reinforcement Learning Agents

### 3.1 PPO (Proximal Policy Optimization)

-   **Formula:**\
    $$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) ] $$

-   **Rationale in HFT:**\
    PPO stabilizes policy updates, making it robust for trading
    environments with noisy reward signals.

------------------------------------------------------------------------

### 3.2 DQN (Deep Q-Network)

-   **Formula:**\
    Bellman update:\
    $$ Q(s,a) = r + \gamma \max_{a'} Q(s', a') $$

-   **Rationale in HFT:**\
    Learns optimal discrete actions (buy, sell, hold) in high-frequency
    environments. Effective with large state spaces.

------------------------------------------------------------------------

### 3.3 SAC (Soft Actor-Critic)

-   **Formula:**\
    Objective combines reward and entropy:\
    $$ J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t) \sim \pi} [r(s_t,a_t) + \alpha H(\pi(\cdot|s_t))] $$

-   **Rationale in HFT:**\
    Balances exploration and exploitation, crucial in dynamic markets
    with changing regimes.

------------------------------------------------------------------------

## 4. LLM-Driven Strategies

### 4.1 Market Sentiment Analysis

-   **Formula:**\
    Use transformer models to map news/text $x_t$ to sentiment score
    $s_t$:\
    $$ s_t = f_{LLM}(x_t) $$

-   **Rationale in HFT:**\
    Captures short-term market impact from news and sentiment. Enables
    event-driven trading.

------------------------------------------------------------------------

### 4.2 News/Event-Driven Trading

-   **Formula:**\
    Signal: trade on event occurrence $E_t$:\
    $$ \text{signal}_t = g(E_t, \Delta P_t) $$

-   **Rationale in HFT:**\
    Events such as earnings or macro releases trigger predictable
    short-term volatility and directional bias.

------------------------------------------------------------------------

### 4.3 Hybrid Decision Models

-   **Formula:**\
    Combine ML signals, RL outputs, and LLM sentiment:\
    $$ \,Signal_t = w_1 f_{ML}(X_t) + w_2 f_{RL}(s_t) + w_3 f_{LLM}(x_t) $$

-   **Rationale in HFT:**\
    Integrating heterogeneous signals allows more robust decision-making
    under uncertainty.

------------------------------------------------------------------------

## 5. Summary

Each method leverages different mathematical principles:\
- **Traditional strategies**: rely on statistical properties of prices.\
- **Machine learning models**: use supervised learning to predict
returns.\
- **Reinforcement learning**: optimizes sequential decision-making under
uncertainty.\
- **LLM-driven methods**: incorporate unstructured data such as news.

Together, they form a **unified multi-strategy HFT system** that can
adapt to diverse market conditions.
