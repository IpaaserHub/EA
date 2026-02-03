# Self-Improving Parameter Optimizer Design

> **Date:** 2026-02-03
> **Status:** Approved for Implementation
> **Author:** AI-assisted brainstorming session

---

## Overview

Build a fully automatic self-improvement layer on top of the existing AI-EA v10.7 trading system. The system will optimize trading parameters (ADX threshold, slope threshold, TP/SL multipliers, etc.) by running backtests, analyzing results with AI, and applying small incremental adjustments.

### Key Decisions Made

| Decision | Choice |
|----------|--------|
| Automation Level | Fully Automatic |
| Scope | Optimize existing v10.7 system (not rebuild) |
| AI Change Strategy | Small adjustments only (±10-20% per cycle) |
| Notifications | Log file only |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    THE OPTIMIZATION LOOP                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │  1. BACKTEST │───▶│  2. ANALYZE  │───▶│  3. SUGGEST  │  │
│   │              │    │              │    │              │  │
│   │ Run trades   │    │ AI looks at  │    │ AI proposes  │  │
│   │ on history   │    │ what worked  │    │ new params   │  │
│   └──────────────┘    └──────────────┘    └──────────────┘  │
│          ▲                                       │          │
│          │            ┌──────────────┐           │          │
│          │            │  4. APPLY    │           │          │
│          └────────────│              │◀──────────┘          │
│                       │ Update the   │                      │
│                       │ config file  │                      │
│                       └──────────────┘                      │
│                              │                              │
│                              ▼                              │
│                       ┌──────────────┐                      │
│                       │  5. REPEAT   │                      │
│                       │  (next day)  │                      │
│                       └──────────────┘                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Why this works (and why it's not "slow AI"):**
- AI is only called once per day (not per trade)
- Actual trading uses fast rule-based code (milliseconds)
- AI does the thinking offline about how to improve

---

## Components to Build

```
/EA/
├── main_genai_custom.py      # Existing - does the actual trading
├── test_historical.py         # Existing - runs backtests
│
├── optimizer/                 # NEW - The self-improvement system
│   ├── __init__.py
│   ├── runner.py             # Runs the optimization loop
│   ├── backtest_runner.py    # Executes backtests, collects results
│   ├── ai_analyzer.py        # Sends results to AI, gets suggestions
│   ├── param_manager.py      # Reads/writes parameter configs
│   └── scheduler.py          # Runs optimization daily (or on schedule)
│
├── config/                    # NEW - Separate config from code
│   ├── params_XAUJPY.json    # Parameters per symbol
│   ├── params_BTCJPY.json
│   └── optimization_log.json # History of all changes
│
└── run_optimizer.py          # NEW - Entry point to start the loop
```

### Component Responsibilities

| Component | Purpose |
|-----------|---------|
| `backtest_runner.py` | Runs `test_historical.py` with current params, captures win rate, profit, etc. |
| `ai_analyzer.py` | Formats results into a prompt, asks AI "what should we change?", parses response |
| `param_manager.py` | Loads params from JSON, applies AI suggestions, saves new params |
| `scheduler.py` | Runs the loop automatically (e.g., every night at 2am) |
| `optimization_log.json` | Keeps history so we can see what worked and rollback if needed |

---

## Technology Stack

### Open Source Libraries to Use

All libraries below have **MIT or Apache 2.0 licenses** - safe for commercial use and selling.

| Component | Library | License | Purpose | Install |
|-----------|---------|---------|---------|---------|
| **Parameter Optimization** | [Optuna](https://optuna.org/) | MIT | Smart parameter search, pruning bad trials | `pip install optuna` |
| **Technical Indicators** | [pandas-ta](https://github.com/twopirllc/pandas-ta) | MIT | Calculate RSI, ADX, ATR, etc. (200+ indicators) | `pip install pandas-ta` |
| **Backtesting** | [bt](https://pmorissette.github.io/bt/) | MIT | Run simulated trades on historical data | `pip install bt` |
| **Scheduling** | [APScheduler](https://github.com/agronholm/apscheduler) | MIT | Run optimizer on schedule (daily, hourly) | `pip install apscheduler` |
| **API Server** | FastAPI | MIT | Already using - keep it | (installed) |
| **Database** | SQLite | Public Domain | Already using - keep it | (built-in) |

### Alternative Options

| Component | Alternative | License | When to Use |
|-----------|-------------|---------|-------------|
| Backtesting | [PyAlgoTrade](https://gbeced.github.io/pyalgotrade/) | Apache 2.0 | More features, better docs |
| Indicators | [talipp](https://pypi.org/project/talipp/) | MIT | Real-time streaming data |
| Indicators | [streaming-indicators](https://pypi.org/project/streaming-indicators/) | MIT | Live tick-by-tick updates |

### Libraries to AVOID (License Issues)

| Library | License | Problem |
|---------|---------|---------|
| Backtesting.py | AGPL-3.0 | Must open-source your entire app |
| QuantConnect Lean | Apache 2.0 | License OK, but it's C# not Python |

### What We Build vs. What We Use

```
┌─────────────────────────────────────────────────────────────────┐
│                    APP ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │         OUR ORIGINAL CODE (competitive advantage)       │    │
│  │                                                         │    │
│  │  • AI Analyzer (prompts, parsing, decision logic)      │    │
│  │  • Self-improvement loop orchestration                  │    │
│  │  • Safety mechanisms (rollback, limits, validation)    │    │
│  │  • Integration with MetaTrader 5                        │    │
│  │  • Unique trading strategy logic                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │         OPEN SOURCE LIBRARIES (free, legal to sell)     │    │
│  │                                                         │    │
│  │  • Optuna → parameter optimization engine               │    │
│  │  • pandas-ta → calculate RSI, ADX, ATR indicators       │    │
│  │  • bt → run backtests on historical data                │    │
│  │  • APScheduler → run optimizer on schedule              │    │
│  │  • FastAPI → API server for MT5 communication           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Example Integration

```python
# How the libraries work together

from optuna import create_study, Trial      # MIT - parameter optimization
import pandas_ta as ta                       # MIT - technical indicators
from apscheduler.schedulers.background import BackgroundScheduler  # MIT
import bt                                    # MIT - backtesting

# OUR CODE: The AI-powered optimizer
class SelfImprovingOptimizer:
    def __init__(self):
        self.study = create_study(direction="maximize")
        self.scheduler = BackgroundScheduler()

    def optimize(self, trial: Trial):
        # Optuna suggests parameters
        params = {
            "adx_threshold": trial.suggest_int("adx_threshold", 3, 30),
            "tp_mult": trial.suggest_float("tp_mult", 1.0, 4.0),
            "sl_mult": trial.suggest_float("sl_mult", 0.8, 3.0),
        }

        # Run backtest with bt library
        profit = self.run_backtest(params)

        # OUR SECRET SAUCE: AI analyzes results
        ai_feedback = self.ai_analyzer.analyze(params, profit)

        return profit

    def run_backtest(self, params):
        # Use bt library for backtesting
        # Use pandas-ta for indicator calculations
        pass

    def start_scheduled_optimization(self):
        # Run optimization every day at 2am
        self.scheduler.add_job(
            self.optimize,
            'cron',
            hour=2
        )
        self.scheduler.start()
```

### Installation Command

```bash
pip install optuna pandas-ta bt apscheduler
```

---

## Parameters to Optimize

These are the parameters the AI will tune:

```python
"adx_threshold": 5,        # How strong must the trend be to trade?
"slope_threshold": 0.00001, # How steep must the price direction be?
"buy_position": 0.50,       # Where in the price range do we buy?
"sell_position": 0.50,      # Where in the price range do we sell?
"tp_mult": 2.0,             # Take Profit = ATR × this number
"sl_mult": 1.5,             # Stop Loss = ATR × this number
```

---

## AI Prompt Design

### Input to AI

```
=== OPTIMIZATION REPORT: XAUJPY ===
Period: 2026-01-01 to 2026-02-01
Current Parameters:
  - adx_threshold: 5
  - slope_threshold: 0.00001
  - tp_mult: 2.0
  - sl_mult: 1.5

Results:
  - Total Trades: 847
  - Win Rate: 52.3%
  - Profit Factor: 1.41
  - Total Profit: ¥234,500
  - Average Win: ¥1,200
  - Average Loss: ¥980
  - Largest Drawdown: ¥45,000

Failed Trade Analysis:
  - 68% of losses happened when ADX < 8
  - 45% of losses happened during low volatility (ATR < 50)
  - Winning trades held average 2.3 hours
  - Losing trades held average 4.1 hours

CONSTRAINTS:
  - Each parameter can only change by ±20% maximum
  - adx_threshold must stay between 3 and 30
  - tp_mult must stay between 1.0 and 4.0
```

### Expected AI Output

```json
{
  "analysis": "Most losses occur when ADX is below 8, suggesting weak trends. Raising threshold will filter these out.",
  "changes": {
    "adx_threshold": {"old": 5, "new": 6, "reason": "Filter weak trend trades"},
    "sl_mult": {"old": 1.5, "new": 1.6, "reason": "Give trades more room"}
  },
  "expected_impact": "Fewer trades but higher win rate"
}
```

---

## Safety Mechanisms

### 1. Parameter Boundaries (Hard Limits)

```python
PARAM_LIMITS = {
    "adx_threshold":    {"min": 3,    "max": 30,   "step": 1},
    "slope_threshold":  {"min": 0.000001, "max": 0.0001, "step": 0.000005},
    "tp_mult":          {"min": 1.0,  "max": 4.0,  "step": 0.1},
    "sl_mult":          {"min": 0.8,  "max": 3.0,  "step": 0.1},
    "buy_position":     {"min": 0.3,  "max": 0.7,  "step": 0.05},
    "sell_position":    {"min": 0.3,  "max": 0.7,  "step": 0.05},
}
```

AI cannot suggest values outside these ranges.

### 2. Rollback System

```
optimization_log.json:
[
  {"date": "2026-02-01", "params": {...}, "profit": 234500, "win_rate": 52.3},
  {"date": "2026-02-02", "params": {...}, "profit": 198000, "win_rate": 48.1},  ← worse!
  {"date": "2026-02-03", "params": {...}, "profit": 256000, "win_rate": 54.2}
]
```

**Auto-rollback rule:** If profit drops more than 30% for 3 consecutive days, revert to the last "good" parameters.

### 3. Validation Before Apply

Before applying new parameters:
1. Run a quick backtest with the proposed changes
2. Compare results to current parameters
3. Only apply if new params are at least as good on test data

### 4. Logging

All optimization changes are logged with timestamps, before/after values, and reasoning.

---

## Implementation Phases

| Phase | What | Deliverable |
|-------|------|-------------|
| **Phase 1** | Extract params to JSON config | `config/params_*.json` files |
| **Phase 2** | Build backtest runner | `optimizer/backtest_runner.py` |
| **Phase 3** | Build AI analyzer | `optimizer/ai_analyzer.py` |
| **Phase 4** | Build param manager | `optimizer/param_manager.py` |
| **Phase 5** | Build main loop | `optimizer/runner.py` |
| **Phase 6** | Add scheduler | `optimizer/scheduler.py` |
| **Phase 7** | Testing & tuning | Validate on historical data |

---

## Business Considerations

### Parameter Decay Problem

Parameters become less effective over time due to:
- Market regime changes (trending vs. ranging)
- Volatility shifts
- Economic events
- Other traders adapting

**Solution:** The self-improving system continuously adapts to changing conditions.

### Recommended Business Model

Sell EA with optimization subscription - the self-improving feature is the competitive advantage that keeps customers paying and the EA profitable.

---

## Learning Resources

### Books
- **Machine Learning for Algorithmic Trading** by Stefan Jansen - covers ML workflow, backtesting, alpha factors
- **Algorithmic Trading Methods** by Robert Kissell - optimization and ML for parameter tuning

### Papers
- [Hyperparameters in RL and How to Tune Them](https://arxiv.org/abs/2306.01324) - hyperparameter optimization theory

### Open Source Examples
- [ImpulseCorp LiveAlgos](https://github.com/impulsecorp/livealgos) - 11,000+ parameter combinations, genetic algorithms
- [QuantConnect Lean](https://github.com/QuantConnect/Lean) - professional backtesting engine
- [ML for Algorithmic Trading Code](https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition)

### Articles
- [Regime-Adaptive Trading Python Guide](https://blog.quantinsti.com/regime-adaptive-trading-python/)
- [Autoregressive Drift Detection Method](https://blog.quantinsti.com/autoregressive-drift-detection-method/)

---

## Trading Concepts Reference

For team members new to trading:

| Term | Meaning |
|------|---------|
| Entry | When we open a trade (BUY or SELL) |
| Exit | When we close the trade |
| Stop Loss (SL) | Auto-close if price goes against us by X amount |
| Take Profit (TP) | Auto-close if price goes in our favor by Y amount |
| ATR | Average True Range - measures volatility |
| RSI | Relative Strength Index - overbought/oversold indicator |
| ADX | Average Directional Index - trend strength (>25 = strong trend) |
| Win Rate | Percentage of trades that are profitable |
| Profit Factor | Gross profit / Gross loss (>1.0 = profitable) |
| Drawdown | Largest peak-to-trough decline |
