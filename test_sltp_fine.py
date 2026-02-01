"""
SL/TP微調整テスト
SLを少し広げてノイズ耐性を上げる
"""

import random
from test_historical import (
    load_mt5_csv, calculate_rsi, calculate_slope,
    calculate_adx, calculate_position,
    SYMBOL_CONFIGS
)

# 微調整パターン（SLを少し広げる方向）
SLTP_PATTERNS = {
    "現在(TP2.0/SL1.2)": {"tp_mult": 2.0, "sl_mult": 1.2},
    "SL+0.3(TP2.0/SL1.5)": {"tp_mult": 2.0, "sl_mult": 1.5},
    "SL+0.6(TP2.0/SL1.8)": {"tp_mult": 2.0, "sl_mult": 1.8},
    "SL+0.8(TP2.0/SL2.0)": {"tp_mult": 2.0, "sl_mult": 2.0},
    "TP少し狭め(TP1.8/SL1.5)": {"tp_mult": 1.8, "sl_mult": 1.5},
    "RR=1(TP1.5/SL1.5)": {"tp_mult": 1.5, "sl_mult": 1.5},
}

# エントリー条件
ENTRY_PARAMS = {
    "adx_threshold": 5,
    "slope_threshold": 0.00001,
    "buy_position": 0.50,
    "sell_position": 0.50,
    "rsi_buy_max": 75,
    "rsi_sell_min": 25,
}


def find_signals(prices: list) -> list:
    """エントリーシグナル検出"""
    signals = []

    for i in range(100, len(prices) - 100):
        history = prices[i-100:i+1]

        rsi = calculate_rsi(history)
        slope = calculate_slope(history, period=50)
        adx = calculate_adx(history)
        position = calculate_position(history)

        if adx < ENTRY_PARAMS["adx_threshold"]:
            continue
        if abs(slope) < ENTRY_PARAMS["slope_threshold"]:
            continue

        if (slope > ENTRY_PARAMS["slope_threshold"] and
            position < ENTRY_PARAMS["buy_position"] and
            rsi < ENTRY_PARAMS["rsi_buy_max"]):
            signals.append({"idx": i, "type": "BUY", "adx": adx})

        elif (slope < -ENTRY_PARAMS["slope_threshold"] and
              position > ENTRY_PARAMS["sell_position"] and
              rsi > ENTRY_PARAMS["rsi_sell_min"]):
            signals.append({"idx": i, "type": "SELL", "adx": adx})

    return signals


def simulate_trade(symbol: str, prices: list, entry_idx: int,
                   trade_type: str, tp_mult: float, sl_mult: float) -> dict:
    """トレードシミュレーション"""
    config = SYMBOL_CONFIGS.get(symbol, SYMBOL_CONFIGS["USDJPY"])
    entry_price = prices[entry_idx]

    # ATR計算
    lookback = min(20, entry_idx)
    if lookback < 5:
        atr = entry_price * 0.001
    else:
        recent = prices[entry_idx - lookback:entry_idx + 1]
        diffs = [abs(recent[i] - recent[i-1]) for i in range(1, len(recent))]
        atr = sum(diffs) / len(diffs) if diffs else entry_price * 0.001

    tp_distance = atr * tp_mult
    sl_distance = atr * sl_mult

    if trade_type == "BUY":
        tp_price = entry_price + tp_distance
        sl_price = entry_price - sl_distance
    else:
        tp_price = entry_price - tp_distance
        sl_price = entry_price + sl_distance

    result = "TIMEOUT"
    exit_price = prices[-1] if entry_idx < len(prices) - 1 else entry_price

    for j in range(entry_idx + 1, min(entry_idx + 100, len(prices))):
        current_price = prices[j]

        if trade_type == "BUY":
            if current_price >= tp_price:
                result = "TP"
                exit_price = tp_price
                break
            elif current_price <= sl_price:
                result = "SL"
                exit_price = sl_price
                break
        else:
            if current_price <= tp_price:
                result = "TP"
                exit_price = tp_price
                break
            elif current_price >= sl_price:
                result = "SL"
                exit_price = sl_price
                break
        exit_price = current_price

    if trade_type == "BUY":
        pips = (exit_price - entry_price) / config["pip_value"]
    else:
        pips = (entry_price - exit_price) / config["pip_value"]

    profit = pips * config["lot_profit_per_pip"]

    return {"result": result, "profit": profit, "pips": pips}


def run_fine_tuning(symbol: str = "XAUJPY"):
    """微調整テスト"""

    print("=" * 70)
    print("[SL/TP微調整テスト] SLを広げてノイズ耐性UP")
    print("=" * 70)

    prices = load_mt5_csv(symbol)
    if not prices:
        print("データ読み込み失敗")
        return

    print(f"  データ本数: {len(prices)}")

    signals = find_signals(prices)
    print(f"  シグナル数: {len(signals)}")
    print("=" * 70)

    results_summary = []

    for pattern_name, params in SLTP_PATTERNS.items():
        trade_results = []
        for signal in signals:
            result = simulate_trade(
                symbol, prices, signal["idx"], signal["type"],
                params["tp_mult"], params["sl_mult"]
            )
            trade_results.append(result)

        wins = [r for r in trade_results if r["profit"] > 0]
        losses = [r for r in trade_results if r["profit"] <= 0]
        total_profit = sum(r["profit"] for r in trade_results)
        total_wins = sum(r["profit"] for r in wins) if wins else 0
        total_losses = abs(sum(r["profit"] for r in losses)) if losses else 0.01
        pf = total_wins / total_losses if total_losses > 0 else 0
        win_rate = len(wins) / len(trade_results) * 100 if trade_results else 0

        tp_count = len([r for r in trade_results if r["result"] == "TP"])
        sl_count = len([r for r in trade_results if r["result"] == "SL"])

        results_summary.append({
            "pattern": pattern_name,
            "win_rate": win_rate,
            "pf": pf,
            "total_profit": total_profit,
            "tp": tp_count,
            "sl": sl_count,
            "tp_mult": params["tp_mult"],
            "sl_mult": params["sl_mult"],
        })

    # 結果表示
    print(f"\n{'パターン':<24} {'勝率':<8} {'PF':<6} {'損益':<12} {'TP/SL比'}")
    print("-" * 70)

    for r in results_summary:
        ratio = r["tp"] / r["sl"] if r["sl"] > 0 else float('inf')
        print(f"{r['pattern']:<24} {r['win_rate']:>5.1f}% {r['pf']:>5.2f} {r['total_profit']:>+10,.0f} {ratio:.2f} ({r['tp']}/{r['sl']})")

    print("=" * 70)

    # 最適解を提案
    # 勝率50%以上かつPF1.3以上を優先
    good_options = [r for r in results_summary if r['win_rate'] >= 50 and r['pf'] >= 1.3]
    if good_options:
        best = max(good_options, key=lambda x: x['total_profit'])
        print(f"\n[推奨] {best['pattern']}")
        print(f"  勝率: {best['win_rate']:.1f}%")
        print(f"  PF: {best['pf']:.2f}")
        print(f"  損益: {best['total_profit']:+,.0f} JPY")
    else:
        best = max(results_summary, key=lambda x: x['total_profit'])
        print(f"\n[最高利益] {best['pattern']}")
        print(f"  損益: {best['total_profit']:+,.0f} JPY、勝率 {best['win_rate']:.1f}%、PF {best['pf']:.2f}")

    return results_summary


if __name__ == "__main__":
    run_fine_tuning()
