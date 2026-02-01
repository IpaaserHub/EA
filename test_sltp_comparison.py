"""
SL/TP比率比較テスト
取引頻度を維持しながら勝率を上げる設定を検証
"""

import random
from test_historical import (
    load_mt5_csv, calculate_rsi, calculate_slope,
    calculate_adx, calculate_position,
    SYMBOL_CONFIGS
)

# テストするSL/TP パターン
SLTP_PATTERNS = {
    "現在(TP2.0/SL1.2)": {
        "tp_mult": 2.0,
        "sl_mult": 1.2,
    },
    "バランス(TP1.5/SL1.5)": {
        "tp_mult": 1.5,
        "sl_mult": 1.5,
    },
    "高勝率(TP1.0/SL2.0)": {
        "tp_mult": 1.0,
        "sl_mult": 2.0,
    },
    "超高勝率(TP0.8/SL2.5)": {
        "tp_mult": 0.8,
        "sl_mult": 2.5,
    },
}

# エントリー条件（v10.6 AGGRESSIVE）
ENTRY_PARAMS = {
    "adx_threshold": 5,
    "slope_threshold": 0.00001,
    "buy_position": 0.50,
    "sell_position": 0.50,
    "rsi_buy_max": 75,
    "rsi_sell_min": 25,
}


def find_signals(prices: list) -> list:
    """エントリーシグナル検出（v10.6設定）"""
    signals = []

    for i in range(100, len(prices) - 100):
        history = prices[i-100:i+1]

        rsi = calculate_rsi(history)
        slope = calculate_slope(history, period=50)
        adx = calculate_adx(history)
        position = calculate_position(history)

        # ADXフィルター
        if adx < ENTRY_PARAMS["adx_threshold"]:
            continue

        # Slopeフィルター
        if abs(slope) < ENTRY_PARAMS["slope_threshold"]:
            continue

        # BUY条件
        if (slope > ENTRY_PARAMS["slope_threshold"] and
            position < ENTRY_PARAMS["buy_position"] and
            rsi < ENTRY_PARAMS["rsi_buy_max"]):
            signals.append({
                "idx": i,
                "type": "BUY",
                "rsi": rsi,
                "slope": slope,
                "adx": adx,
                "position": position
            })

        # SELL条件
        elif (slope < -ENTRY_PARAMS["slope_threshold"] and
              position > ENTRY_PARAMS["sell_position"] and
              rsi > ENTRY_PARAMS["rsi_sell_min"]):
            signals.append({
                "idx": i,
                "type": "SELL",
                "rsi": rsi,
                "slope": slope,
                "adx": adx,
                "position": position
            })

    return signals


def simulate_trade_with_sltp(symbol: str, prices: list, entry_idx: int,
                              trade_type: str, tp_mult: float, sl_mult: float,
                              lot_multiplier: float = 1.0) -> dict:
    """指定SL/TPでトレードシミュレーション"""
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

    # SL/TP計算
    tp_distance = atr * tp_mult
    sl_distance = atr * sl_mult

    if trade_type == "BUY":
        tp_price = entry_price + tp_distance
        sl_price = entry_price - sl_distance
    else:
        tp_price = entry_price - tp_distance
        sl_price = entry_price + sl_distance

    # シミュレーション（最大100本）
    result = "TIMEOUT"
    exit_price = prices[-1] if entry_idx < len(prices) - 1 else entry_price
    bars_held = 0

    for j in range(entry_idx + 1, min(entry_idx + 100, len(prices))):
        current_price = prices[j]
        bars_held = j - entry_idx

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

    # 損益計算
    if trade_type == "BUY":
        pips = (exit_price - entry_price) / config["pip_value"]
    else:
        pips = (entry_price - exit_price) / config["pip_value"]

    profit = pips * config["lot_profit_per_pip"] * lot_multiplier

    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "result": result,
        "profit": profit,
        "pips": pips,
        "bars_held": bars_held
    }


def run_comparison(symbol: str = "XAUJPY", max_trades: int = 200, deposit: int = 50000):
    """SL/TP比較テスト実行"""

    print("=" * 70)
    print("[SL/TP比率比較テスト] 取引頻度維持 × 勝率向上")
    print("=" * 70)
    print(f"  シンボル: {symbol}")
    print(f"  最大取引数: {max_trades}")
    print(f"  入金額: {deposit:,} JPY")
    print("=" * 70)

    # データ読み込み
    prices = load_mt5_csv(symbol)
    if not prices:
        print("データ読み込み失敗")
        return

    print(f"  データ本数: {len(prices)}")

    # シグナル検出（全パターン共通）
    signals = find_signals(prices)
    print(f"  検出シグナル数: {len(signals)}")
    print("=" * 70)

    if not signals:
        print("シグナルなし")
        return

    # ランダムサンプリング
    random.seed(42)
    test_signals = signals[:max_trades] if len(signals) <= max_trades else random.sample(signals, max_trades)

    lot_multiplier = deposit / 50000.0
    results_summary = []

    for pattern_name, params in SLTP_PATTERNS.items():
        print(f"\n[{pattern_name}]")
        print(f"  TP: {params['tp_mult']}×ATR / SL: {params['sl_mult']}×ATR")

        # シミュレーション
        trade_results = []
        for signal in test_signals:
            result = simulate_trade_with_sltp(
                symbol, prices, signal["idx"], signal["type"],
                params["tp_mult"], params["sl_mult"], lot_multiplier
            )
            result["symbol"] = symbol
            result["position_type"] = signal["type"]
            trade_results.append(result)

        # 集計
        wins = [r for r in trade_results if r["profit"] > 0]
        losses = [r for r in trade_results if r["profit"] <= 0]
        total_profit = sum(r["profit"] for r in trade_results)
        total_wins = sum(r["profit"] for r in wins) if wins else 0
        total_losses = abs(sum(r["profit"] for r in losses)) if losses else 0.01
        pf = total_wins / total_losses if total_losses > 0 else float('inf')
        win_rate = len(wins) / len(trade_results) * 100 if trade_results else 0

        # 結果種別の内訳
        tp_count = len([r for r in trade_results if r["result"] == "TP"])
        sl_count = len([r for r in trade_results if r["result"] == "SL"])
        timeout_count = len([r for r in trade_results if r["result"] == "TIMEOUT"])

        print(f"  取引数: {len(trade_results)}")
        print(f"  勝率: {win_rate:.1f}% ({len(wins)}/{len(trade_results)})")
        print(f"  内訳: TP={tp_count}, SL={sl_count}, TIMEOUT={timeout_count}")
        print(f"  PF: {pf:.2f}")
        print(f"  損益: {total_profit:+,.0f} JPY")

        # 期待値計算
        avg_win = total_wins / len(wins) if wins else 0
        avg_loss = total_losses / len(losses) if losses else 0
        expected_value = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
        print(f"  期待値: {expected_value:+,.0f} JPY/trade")

        results_summary.append({
            "pattern": pattern_name,
            "trades": len(trade_results),
            "wins": len(wins),
            "win_rate": win_rate,
            "pf": pf,
            "total_profit": total_profit,
            "expected_value": expected_value,
            "tp_mult": params["tp_mult"],
            "sl_mult": params["sl_mult"],
        })

    # 比較サマリー
    print("\n" + "=" * 70)
    print("[比較サマリー]")
    print("=" * 70)
    print(f"{'パターン':<22} {'勝率':<10} {'PF':<8} {'損益':<14} {'期待値':<10}")
    print("-" * 70)

    for r in results_summary:
        pf_str = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "∞"
        print(f"{r['pattern']:<22} {r['win_rate']:.1f}%{'':<5} {pf_str:<8} {r['total_profit']:+,.0f} JPY{'':<3} {r['expected_value']:+,.0f}")

    print("=" * 70)

    # 推奨
    best = max(results_summary, key=lambda x: x['total_profit'] if x['pf'] > 0.9 else -999999)
    print(f"\n[推奨] {best['pattern']}")
    print(f"  理由: 利益 {best['total_profit']:+,.0f} JPY、勝率 {best['win_rate']:.1f}%、PF {best['pf']:.2f}")
    print(f"  設定: TP={best['tp_mult']}×ATR, SL={best['sl_mult']}×ATR")

    return results_summary


if __name__ == "__main__":
    run_comparison(symbol="XAUJPY", max_trades=200, deposit=50000)
