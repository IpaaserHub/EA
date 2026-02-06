# AI Features Master Plan
## 5 Features to Make the EA Product AI-Powered

Date: 2026-02-06

---

## Priority Order & Dependencies

```
1. Multi-Timeframe Confirmation (no deps, rule-based, highest bang-for-buck)
2. Adaptive Exit Timing (no deps, extends backtest engine)
3. Market Regime Detection (needs 1 & 2 for full value, but can be built independently)
4. News/Event Filter (independent, needs economic calendar data)
5. AI Optimization Reports (independent, needs OpenAI API)
```

Features 1-3 improve trading performance. Features 4-5 improve product value/safety.

---

## Feature 1: Multi-Timeframe Confirmation

**Impact: HIGH | Effort: 2-3 days | Risk: LOW**

### What
Aggregate H1 candles to H4/D1, compute indicators on higher timeframes, only take H1 signals when H4/D1 trend agrees.

### Files to Create/Modify
- `src/backtest/indicators.py` — Add `aggregate_candles()` function
- `src/backtest/engine.py` — Add `_check_entry_mtf()`, add `use_mtf` param
- `src/config/param_manager.py` — Add MTF params to `PARAM_LIMITS`
- `main_genai_custom.py` — Add H4/D1 filtering in `analyze_market_logic()`
- `tests/test_mtf.py` — New test file

### New Parameters
```python
"h4_adx_min": {"min": 10, "max": 25, "type": "int"},
"h4_slope_filter": {"min": 0, "max": 1, "type": "int"},  # 0=off, 1=on
"d1_slope_filter": {"min": 0, "max": 1, "type": "int"},
```

### Key Rules
- Only use COMPLETED higher-TF candles (no repainting)
- Graceful degradation: if not enough H4/D1 data, skip filter
- H4 slope must agree with H1 signal direction
- D1 slope provides bias (soft filter for AGGRESSIVE, hard for STABLE)
- In backtest: use `prices[:i]` for aggregation (no look-ahead bias)

### Expected Impact
- 30-40% fewer trades with H4 filter
- +5-10% win rate improvement
- +0.15-0.4 profit factor improvement

---

## Feature 2: Adaptive Exit Timing

**Impact: HIGH | Effort: 3-4 days | Risk: LOW**

### What
Replace fixed TP/SL with ATR-adaptive trailing stop. Once trade moves X ATR in profit, activate a trailing stop that adapts to current volatility.

### Files to Modify
- `src/backtest/engine.py` — Modify `Trade` dataclass, modify `_check_exit()`
- `src/config/param_manager.py` — Add trailing stop params
- `main_genai_custom.py` — Add trailing logic to `/check_exit` endpoint
- `tests/test_adaptive_exit.py` — New test file

### New Parameters
```python
"trail_activation_atr": {"min": 0.5, "max": 2.0, "type": "float"},
"trail_distance_mult": {"min": 0.5, "max": 3.0, "type": "float"},
"max_bars_in_trade": {"min": 10, "max": 200, "type": "int"},
```

### Key Rules
- Trailing stop only moves in favorable direction (never retreats)
- Breakeven stop: move SL to entry when profit >= 0.5 ATR
- Time exit: close trade after max_bars_in_trade candles
- Hard SL always respected as safety net
- Trail distance adapts to CURRENT ATR (not entry ATR)

### Implementation Pattern
```python
# In _check_exit:
1. Track highest_profit (BUY) or lowest_profit (SELL) per trade
2. Once unrealized >= trail_activation_atr * current_atr: activate trail
3. Trail = peak - (trail_distance_mult * current_atr)
4. Exit priority: TP hit > trailing stop hit > time limit > hard SL hit
```

### Expected Impact
- Captures larger trends (current fixed TP cuts winners short)
- Reduces drawdown on reversal trades
- Time exit prevents capital lock-up in sideways markets

---

## Feature 3: Market Regime Detection

**Impact: HIGH | Effort: 1-2 weeks | Risk: MEDIUM**

### What
Unsupervised clustering (KMeans or GMM) on market features to classify regimes (trending, ranging, volatile, calm). Select pre-optimized parameter sets per regime.

### Files to Create
- `src/ai/regime_detector.py` — New module with RegimeDetector class
- `src/ai/__init__.py` — Package init
- `config/regime_params/` — Per-regime parameter files
- `tests/test_regime_detector.py` — Tests

### Files to Modify
- `src/backtest/engine.py` — Accept regime-based param switching
- `main_genai_custom.py` — Add regime detection before entry logic
- `requirements.txt` — No new deps needed (sklearn already present)

### Approach
```
1. Features: ADX, ATR percentile, slope magnitude, Bollinger bandwidth
2. Model: KMeans with n_clusters=3 (trending, ranging, volatile)
3. Train: Fit on historical data, label clusters by characteristics
4. Use: Classify current window, select matching param set
5. Validate: Walk-forward test each regime's params separately
```

### Why 3 Regimes (not 4)
- Trending: High ADX, consistent slope → use trend-following params
- Ranging: Low ADX, low slope → either skip trading or use mean-reversion
- Volatile: High ATR, unstable direction → tighten stops, reduce position

### Expected Impact
- Avoids using trend params in ranging markets (biggest loss source)
- Adapts to market conditions between daily optimizer runs

---

## Feature 4: News/Event Filter

**Impact: MEDIUM | Effort: 2-3 days | Risk: LOW**

### What
Pause trading before/after high-impact economic events (NFP, FOMC, BOJ) to prevent trades from being blown out by news spikes.

### Files to Create
- `src/ai/news_filter.py` — NewsFilter class
- `config/economic_calendar.json` — Static calendar of major events
- `tests/test_news_filter.py` — Tests

### Files to Modify
- `main_genai_custom.py` — Add news check in `analyze_market_logic()` after session filter

### Approach: Static Calendar First
```json
{
  "events": [
    {"name": "NFP", "currency": "USD", "schedule": "first_friday_monthly",
     "time_utc": "13:30", "pause_before_min": 60, "pause_after_min": 120},
    {"name": "FOMC", "currency": "USD", "dates_2026": ["2026-01-29", ...],
     "time_utc": "19:00", "pause_before_min": 60, "pause_after_min": 120},
    {"name": "BOJ Decision", "currency": "JPY", "dates_2026": ["2026-01-24", ...],
     "time_utc": "03:00", "pause_before_min": 60, "pause_after_min": 60}
  ]
}
```

### Pause Windows
| Event Type | Before | After |
|-----------|--------|-------|
| NFP/FOMC | 60 min | 120 min |
| CPI/GDP | 30 min | 60 min |
| BOJ Decision | 60 min | 60 min |
| Other High Impact | 30 min | 30 min |

### Integration Point
```python
# In analyze_market_logic(), after line 1368 (session filter), before line 1370:
can_trade_news, news_reason = news_filter.should_trade(symbol)
if not can_trade_news:
    return {"action": "NO_TRADE", "comment": news_reason, ...}
```

### LLM NOT needed here. Rule-based is better because:
- Event impact is already categorized (High/Medium/Low)
- Decision is binary (trade / don't trade)
- Zero cost, zero latency, no hallucination risk

---

## Feature 5: AI Optimization Reports

**Impact: MEDIUM (sales) | Effort: 2-3 days | Risk: LOW**

### What
Generate human-readable Japanese optimization reports using LLM after each optimizer run. Key selling point for the product.

### Files to Create
- `src/optimizer/report_generator.py` — ReportGenerator class
- `tests/test_report_generator.py` — Tests

### Files to Modify
- `src/optimizer/optimization_loop.py` — Call report generator after optimization
- `main_genai_custom.py` — Add `/optimization/report` API endpoint (optional)

### Two-Pass LLM Pattern
1. **Pass 1**: Feed OptimizationRun data → get structured JSON analysis
2. **Pass 2**: Feed JSON analysis → get natural language Japanese report

### Model: gpt-4o-mini
- Cost per report: ~0.2 yen (essentially free)
- Handles Japanese well
- Already integrated in codebase

### Report Sections
1. 概要 (Summary) — 3 lines max
2. パフォーマンス比較 (Performance comparison table)
3. ウォークフォワード検証結果 (Walk-forward results)
4. パラメータ変更の説明 (AI-explained parameter changes)
5. リスク評価 (Risk assessment)
6. 免責事項 (Legal disclaimer — REQUIRED)

### Legal Requirements (from business docs)
- Never say 「必ず儲かる」(guaranteed profit)
- Always include 「過去のデータに基づく結果」(based on historical data)
- Always include full 免責事項 (disclaimer)
- These are HARD-CODED in the template, not left to LLM

---

## Implementation Order

```
Week 1: Feature 1 (MTF) + Feature 2 (Adaptive Exit)
  - These are independent, can be built in parallel
  - Both improve the core trading engine
  - Both can be optimized with existing Optuna + WFO

Week 2: Feature 4 (News Filter) + Feature 5 (AI Reports)
  - Quick wins for product value
  - News filter is simple JSON + rule logic
  - Reports use existing OpenAI integration

Week 3-4: Feature 3 (Regime Detection)
  - Most complex feature
  - Benefits from having MTF and adaptive exits already working
  - Requires training data and careful validation
```
