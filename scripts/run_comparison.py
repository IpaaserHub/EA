"""
Full comparison: Old Code vs New Features on all 3 symbols.
Runs backtest in 4 modes and collects metrics.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtest.engine import BacktestEngine
from backtest.data_loader import load_prices
from ai.regime_detector import RegimeDetector, TRENDING, RANGING, VOLATILE

# Base params (old code defaults)
BASE_PARAMS = {
    "adx_threshold": 5,
    "slope_threshold": 0.00001,
    "buy_position": 0.50,
    "sell_position": 0.50,
    "rsi_buy_max": 75,
    "rsi_sell_min": 25,
    "tp_mult": 2.0,
    "sl_mult": 1.5,
}

# Adaptive exit params (new)
ADAPTIVE_PARAMS = {
    **BASE_PARAMS,
    "trail_activation_atr": 1.0,
    "trail_distance_mult": 1.5,
    "max_bars_in_trade": 48,
}

DATA_FILES = {
    "USDJPY": "data/USDJPYH1.csv",
    "XAUJPY": "data/XAUJPYH1.csv",
    "BTCJPY": "data/BTCJPYH1.csv",
}

def fmt(result):
    d = result.to_dict()
    return {
        "trades": d["total_trades"],
        "wins": d["wins"],
        "losses": d["losses"],
        "win_rate": d["win_rate"],
        "profit_factor": d["profit_factor"],
        "total_profit": d["total_profit"],
        "max_drawdown": d["max_drawdown"],
        "avg_win": d["avg_win"],
        "avg_loss": d["avg_loss"],
    }

results = {}

for symbol, path in DATA_FILES.items():
    prices = load_prices(path)
    if not prices:
        print(f"SKIP {symbol}: no data at {path}")
        continue
    print(f"\n{'='*60}")
    print(f"  {symbol} — {len(prices)} H1 candles")
    print(f"{'='*60}")

    engine = BacktestEngine(prices, lookback=50)
    sym_results = {}

    # Mode 1: Baseline (old code)
    r1 = engine.run(BASE_PARAMS, use_mtf=False)
    sym_results["1_baseline"] = fmt(r1)
    print(f"  Baseline:       {r1.total_trades} trades, WR={r1.win_rate:.1f}%, PF={r1.profit_factor:.2f}, P={r1.total_profit:.1f}")

    # Mode 2: + MTF only
    r2 = engine.run(BASE_PARAMS, use_mtf=True)
    sym_results["2_mtf_only"] = fmt(r2)
    print(f"  +MTF:           {r2.total_trades} trades, WR={r2.win_rate:.1f}%, PF={r2.profit_factor:.2f}, P={r2.total_profit:.1f}")

    # Mode 3: + Adaptive Exit only
    r3 = engine.run(ADAPTIVE_PARAMS, use_mtf=False)
    sym_results["3_adaptive_only"] = fmt(r3)
    print(f"  +Adaptive:      {r3.total_trades} trades, WR={r3.win_rate:.1f}%, PF={r3.profit_factor:.2f}, P={r3.total_profit:.1f}")

    # Mode 4: MTF + Adaptive (all features)
    r4 = engine.run(ADAPTIVE_PARAMS, use_mtf=True)
    sym_results["4_mtf_adaptive"] = fmt(r4)
    print(f"  +MTF+Adaptive:  {r4.total_trades} trades, WR={r4.win_rate:.1f}%, PF={r4.profit_factor:.2f}, P={r4.total_profit:.1f}")

    # Mode 5: Regime Detection
    detector = RegimeDetector(window_size=50)
    detector.fit(prices)
    # Detect regime on last 50 candles
    regime_result = detector.detect(prices[-50:])
    regime_params = detector.get_regime_params(regime_result.regime, ADAPTIVE_PARAMS)
    r5 = engine.run(regime_params, use_mtf=True)
    sym_results["5_regime_adjusted"] = fmt(r5)
    sym_results["detected_regime"] = regime_result.to_dict()
    print(f"  +Regime({regime_result.regime[:4]}): {r5.total_trades} trades, WR={r5.win_rate:.1f}%, PF={r5.profit_factor:.2f}, P={r5.total_profit:.1f}")

    results[symbol] = sym_results

# Save results
with open("scripts/comparison_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n\nResults saved to scripts/comparison_results.json")

# Summary table
print(f"\n{'='*90}")
print(f"  SUMMARY TABLE")
print(f"{'='*90}")
print(f"{'Symbol':<10} {'Mode':<20} {'Trades':>7} {'WinRate':>8} {'PF':>6} {'Profit':>10} {'MaxDD':>10}")
print(f"{'-'*90}")
for symbol, data in results.items():
    for mode_key in ["1_baseline", "2_mtf_only", "3_adaptive_only", "4_mtf_adaptive", "5_regime_adjusted"]:
        m = data[mode_key]
        label = {
            "1_baseline": "Baseline",
            "2_mtf_only": "+MTF",
            "3_adaptive_only": "+AdaptiveExit",
            "4_mtf_adaptive": "+MTF+Adaptive",
            "5_regime_adjusted": "+All+Regime",
        }[mode_key]
        print(f"{symbol:<10} {label:<20} {m['trades']:>7} {m['win_rate']:>7.1f}% {m['profit_factor']:>6.2f} {m['total_profit']:>10.1f} {m['max_drawdown']:>10.1f}")
    print()
