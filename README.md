# Stock Risk Prediction System

## Monte Carlo-Based Stock Price Interval Prediction and Portfolio Risk Simulation

**College-Level Industry-Grade Application**

This application provides probabilistic stock price predictions using Monte Carlo simulation methods with strong statistical foundations. It predicts price intervals (confidence intervals) rather than point estimates, and includes comprehensive risk metrics and stress testing capabilities.

---

## 🎯 Project Overview

### Purpose

- **Predict probabilistic price ranges** (confidence intervals) for any stock
- **Simulate portfolio risk** using Monte Carlo methods
- **Focus on statistical coverage**, not point prediction
- **Target ≥98% empirical coverage** through calibration & backtesting

### Key Features

- ✅ Unlimited historical data fetch (not capped at 7 years)
- ✅ Support for NSE, NYSE, NASDAQ tickers
- ✅ Multiple volatility models (SD, EWMA, GARCH)
- ✅ Monte Carlo simulation with Student-t distributed shocks
- ✅ Regime-aware drift adjustments (Bull/Neutral/Bear)
- ✅ Comprehensive risk metrics (VaR, Expected Shortfall, Sharpe, Sortino)
- ✅ Rolling backtesting with coverage validation
- ✅ Stress testing under extreme scenarios
- ✅ Interactive Streamlit frontend

---

## 📋 Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
See `requirements.txt` for complete list:
- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- yfinance >= 0.2.28
- plotly >= 5.17.0
- scipy >= 1.11.0

---

## 🚀 Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the application:**
   - The app will open in your default web browser
   - Default URL: http://localhost:8501

---

## 📁 Project Structure

```
.
├── app.py                    # Streamlit frontend application
├── requirements.txt          # Python dependencies
├── README.md                # This file
│
├── data/
│   ├── __init__.py
│   └── loader.py            # Data fetching and cleaning
│
├── models/
│   ├── __init__.py
│   ├── returns.py           # Returns calculation (log returns)
│   ├── volatility.py        # Volatility models (SD, EWMA, GARCH)
│   ├── monte_carlo.py       # Monte Carlo simulation engine
│   └── stress.py            # Stress testing scenarios
│
├── backtesting/
│   ├── __init__.py
│   └── coverage.py          # Backtesting and coverage calibration
│
└── utils/
    ├── __init__.py
    └── metrics.py           # Risk metrics (VaR, ES, Sharpe, etc.)
```

---

## 🔬 Statistical Methodology

### Base Model: Geometric Brownian Motion (GBM)

The stock price follows:
```
dS = μS dt + σS dW
```

Where:
- **S**: Stock price
- **μ**: Drift (expected return)
- **σ**: Volatility (standard deviation)
- **dW**: Wiener process (Brownian motion)

### Discrete Form

For simulation:
```
S_{t+1} = S_t * exp((μ - σ²/2) * dt + σ * sqrt(dt) * ε_t)
```

### Enhancements

1. **Student-t Distributed Shocks**
   - Accounts for fat tails (extreme events)
   - Degrees of freedom: 5 (configurable)
   - Normalized to unit variance

2. **Regime-Aware Drift**
   - Classifies market regime: Bull / Neutral / Bear
   - Adjusts drift based on recent returns
   - More realistic in volatile markets

3. **Volatility Scaling**
   - Supports stress testing scenarios
   - Can multiply volatility for extreme events
   - Calibration through backtesting

### Returns Calculation

**Log Returns** (used throughout):
```
r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
```

**Why log returns?**
- Time-additive (multi-period = sum of single-period)
- Symmetric (multiplying by -1 gives inverse)
- Aligns with continuous-time finance theory
- Better statistical properties

### Volatility Models

1. **Standard Deviation (SD)**
   - Rolling standard deviation
   - Window: 252 days (1 year)
   - Simple and robust baseline

2. **EWMA (Exponentially Weighted Moving Average)**
   - Decay factor: λ = 0.94 (RiskMetrics standard)
   - More weight to recent observations
   - More responsive to market changes

3. **GARCH(1,1)** (Optional)
   - Generalized Autoregressive Conditional Heteroskedasticity
   - Allows mean reversion in variance
   - More sophisticated modeling

---

## 📊 Output Metrics

### Price Metrics

- **Mean final price**: Average across all simulations
- **Median final price**: 50th percentile
- **Confidence intervals**: 90%, 95%, 99%
  - Lower and upper bounds
  - Range (difference)

### Risk Metrics

1. **Value at Risk (VaR)**
   - VaR 95%: Maximum expected loss at 95% confidence
   - VaR 99%: Maximum expected loss at 99% confidence
   - Expressed as dollar amount or percentage

2. **Expected Shortfall (Conditional VaR)**
   - Expected loss given that loss exceeds VaR
   - More conservative than VaR
   - Considers tail risk

3. **Probability of Loss**
   - Probability that final price < initial price
   - Percentage (0 to 100%)

4. **Maximum Drawdown**
   - Largest peak-to-trough decline
   - Percentage of peak value

5. **Sharpe Ratio**
   - Risk-adjusted return measure
   - Formula: (Return - Risk-Free Rate) / Volatility
   - Higher is better

6. **Sortino Ratio**
   - Like Sharpe, but only penalizes downside volatility
   - Formula: (Return - Risk-Free Rate) / Downside Deviation
   - More appropriate for asymmetric returns

---

## 🔍 Backtesting & Calibration

### Rolling Backtest

- Uses expanding window (all data up to test point)
- Steps forward quarterly (63 days)
- For each test point:
  1. Train on historical data
  2. Generate prediction interval
  3. Check if actual future price falls within interval
  4. Calculate empirical coverage

### Coverage Calibration

- **Target**: ≥98% coverage for 95% confidence interval
- Adjusts volatility multiplier to achieve target
- Uses binary search over multiplier values
- Validates on out-of-sample data

### Accuracy Definition

- **Accuracy = Interval Coverage**
- Not point prediction accuracy
- Measures how often actual price falls within predicted interval
- Higher coverage = more conservative (wider intervals)

---

## 💥 Stress Testing

Predefined scenarios:

1. **Baseline** (normal conditions)
   - Standard volatility and drift

2. **Moderate Stress** (volatility × 1.5)
   - 50% increase in volatility
   - Tests resilience to moderate shocks

3. **Severe Stress** (volatility × 3)
   - 200% increase in volatility
   - Simulates crash conditions

4. **Negative Drift** (-10% annualized)
   - Negative expected return
   - Bear market scenario

5. **Combined Crash** (volatility × 3, drift -15%)
   - Extreme scenario
   - Combines high volatility and negative drift

---

## 🎨 User Interface

### Sidebar Configuration

- **Stock Ticker**: Enter ticker symbol (e.g., AAPL, MSFT, RELIANCE.NS)
- **Date Range**: Start and end dates for historical data
- **Forecast Horizon**: 1 Day, 1 Month, 3 Months, 6 Months, 1 Year
- **Volatility Model**: SD, EWMA, or GARCH
- **Confidence Levels**: 90%, 95%, 99% (multi-select)
- **Number of Simulations**: 5,000 to 100,000 (slider)
- **Stress Testing**: Enable/disable
- **Backtesting**: Enable/disable

### Main Display

1. **Data Summary**: Trading days, date range, current price
2. **Prediction Results**: Mean, median, confidence intervals
3. **Visualizations**:
   - Histogram of predicted prices
   - Sample simulation paths
   - Confidence bands
4. **Risk Metrics**: All calculated metrics in organized sections
5. **Stress Test Results** (if enabled): Comparison across scenarios
6. **Backtest Results** (if enabled): Coverage analysis and charts

---

## ⚠️ Limitations & Assumptions

### Important Limitations

1. **Probabilistic Predictions**
   - All predictions are probabilistic intervals, not guarantees
   - Past performance does not guarantee future results
   - Model uncertainty is inherent

2. **Model Assumptions**
   - Assumes Geometric Brownian Motion as base model
   - Student-t distribution for shocks (degrees of freedom = 5)
   - Market regime classification based on recent returns
   - Assumes market conditions remain similar to historical period

3. **Data Limitations**
   - Requires minimum 1,000 trading days
   - Does not account for fundamental changes (mergers, splits, etc.)
   - Relies on historical price data only
   - No fundamental analysis included

4. **Statistical Coverage**
   - Target coverage ≥98% for 95% confidence intervals
   - Coverage validated through backtesting
   - Calibration may be necessary for different market conditions
   - Coverage may vary by stock and market conditions

5. **Risk Metrics**
   - VaR and Expected Shortfall are estimates based on simulations
   - May not capture tail events beyond historical experience
   - Stress tests use predefined scenarios, not exhaustive
   - Does not account for liquidity risk, credit risk, etc.

6. **Computational**
   - Backtesting can be slow (several minutes)
   - Large simulation counts (100,000) may take time
   - GARCH model optimization may fail on some datasets

### Use Disclaimer

**This tool is for educational and research purposes only.**

- Do not use for actual investment decisions without professional financial advice
- Not a substitute for professional financial analysis
- No guarantee of accuracy or profitability
- Use at your own risk

---

## 🔧 Technical Details

### Data Ingestion

- **Source**: yfinance (Yahoo Finance API)
- **Supported Exchanges**: NYSE, NASDAQ, NSE (Indian stocks)
- **NSE Format**: Add `.NS` suffix (e.g., `RELIANCE.NS`)
- **Data Cleaning**:
  - Removes rows with Close ≤ 0
  - Removes rows with Volume ≤ 0
  - Removes rows with High < Low
  - Forward-fills missing values only if necessary
  - Removes duplicate dates

### Random Seed

- Default seed: 42 (reproducible results)
- Can be changed in `MonteCarloSimulator` initialization
- Ensures consistent results across runs

### Code Quality

- Type hints throughout
- Comprehensive docstrings
- Modular architecture
- No hardcoded values (except defaults)
- Error handling and validation

---

## 📚 Mathematical Foundations

### Log Returns

Log returns are used because they have superior mathematical properties:
- **Additivity**: Multi-period return = sum of single-period returns
- **Symmetry**: ln(1/r) = -ln(r)
- **Continuity**: Aligns with continuous-time models
- **Normal Distribution**: Log returns more likely to be normally distributed

### Student-t Distribution

Standard normal distribution assumes thin tails. Real stock returns show fat tails (extreme events more common). Student-t distribution with low degrees of freedom (df=5) accounts for this.

**Normalized Student-t**:
```
ε ~ t(df) / sqrt(df / (df - 2))
```

This ensures unit variance while maintaining fat tails.

### Volatility Estimation

**EWMA Variance**:
```
σ²_t = (1-λ)r²_{t-1} + λσ²_{t-1}
```

**GARCH(1,1) Variance**:
```
σ²_t = ω + αε²_{t-1} + βσ²_{t-1}
```

Where: ω > 0, α ≥ 0, β ≥ 0, α + β < 1 (stationarity)

### Monte Carlo Simulation

For each simulation path:
1. Start at initial price S₀
2. For each time step:
   - Generate Student-t shock ε
   - Apply GBM formula: S_{t+1} = S_t * exp((μ - σ²/2)dt + σ√dt * ε)
3. Record final price
4. Repeat for N simulations
5. Calculate statistics from final prices

---

## 🎓 Academic Justification

This application demonstrates:

1. **Probability & Statistics**
   - Confidence intervals and coverage
   - Hypothesis testing (backtesting)
   - Distribution theory (Student-t, Normal)
   - Time series analysis

2. **Financial Engineering**
   - Stochastic processes (GBM)
   - Risk metrics (VaR, Expected Shortfall)
   - Portfolio theory (Sharpe, Sortino)
   - Volatility modeling (EWMA, GARCH)

3. **Computational Finance**
   - Monte Carlo methods
   - Numerical simulation
   - Optimization (GARCH calibration)

4. **Software Engineering**
   - Modular architecture
   - Type safety
   - Error handling
   - User interface design

---

## 🐛 Troubleshooting

### Common Issues

1. **"No data found for ticker"**
   - Check ticker symbol spelling
   - For NSE stocks, use `.NS` suffix
   - Verify ticker exists on Yahoo Finance

2. **"Insufficient data after cleaning"**
   - Ticker may not have enough trading history
   - Try a different ticker or adjust date range
   - Minimum 1,000 trading days required

3. **Backtesting takes too long**
   - Reduce number of simulations
   - Increase step size in backtest
   - Disable backtesting for initial exploration

4. **GARCH model fails**
   - Falls back to EWMA automatically
   - May indicate insufficient data
   - Use EWMA or SD model instead

5. **App crashes or freezes**
   - Reduce number of simulations
   - Check available memory
   - Restart application

---

## 📝 Future Enhancements

Potential improvements (not implemented):

- Portfolio-level analysis (multiple stocks)
- Correlation modeling (multivariate GBM)
- Alternative distributions (skewed-t, GED)
- Real-time data updates
- Export results to CSV/PDF
- More sophisticated regime detection (hidden Markov models)
- Machine learning enhancements (regime classification)
- Option pricing integration
- Sensitivity analysis (Greeks)

---

## 📄 License

This project is provided for educational and research purposes.

---

## 👤 Author

**Quantitative Finance Engineering Team**

Built for college-level academic evaluation with industry-grade methodology.

---

## 🙏 Acknowledgments

- **yfinance**: Yahoo Finance data API
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Academic References**: 
  - RiskMetrics methodology (EWMA)
  - GARCH models (Engle, Bollerslev)
  - Monte Carlo methods in finance
  - Value at Risk theory (Jorion)

---

## 📞 Support

For issues or questions:
1. Check this README
2. Review code comments and docstrings
3. Consult academic finance literature
4. Review error messages for guidance

---

**Last Updated**: 2024

**Version**: 1.0

**Status**: Production-Ready for Academic Use
