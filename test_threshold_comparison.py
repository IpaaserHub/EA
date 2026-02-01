"""
閾値比較テスト
ADX閾値を変更した場合のシグナル数・勝率の変化を検証
"""

import random
from test_historical import (
    load_mt5_csv, calculate_rsi, calculate_slope,
    calculate_adx, calculate_position, simulate_trade,
    SYMBOL_CONFIGS
)

# テストする閾値パターン
THRESHOLD_PATTERNS = {
    "現在(ADX=10)": {
        "adx_threshold": 10,
        "slope_threshold": 0.00002,
        "buy_position": 0.50,
        "sell_position": 0.50,
    },
    "緩和(ADX=5)": {
        "adx_threshold": 5,
        "slope_threshold": 0.00001,
        "buy_position": 0.50,
        "sell_position": 0.50,
    },
    "超緩和(ADX=3)": {
        "adx_threshold": 3,
        "slope_threshold": 0.000005,
        "buy_position": 0.50,
        "sell_position": 0.50,
    },
    "最緩和(ADX=0)": {
        "adx_threshold": 0,  # ADXフィルターなし
        "slope_threshold": 0.000001,
        "buy_position": 0.50,
        "sell_position": 0.50,
    },
}


def find_signals_with_params(prices: list, params: dict) -> list:
    """指定パラメータでシグナル検出"""
    signals = []

    for i in range(100, len(prices) - 100):
        history = prices[i-100:i+1]

        rsi = calculate_rsi(history)
        slope = calculate_slope(history, period=50)
        adx = calculate_adx(history)
        position = calculate_position(history)

        # ADXフィルター
        if adx < params["adx_threshold"]:
            continue

        # Slopeフィルター
        if abs(slope) < params["slope_threshold"]:
            continue

        # BUY条件
        if (slope > params["slope_threshold"] and
            position < params["buy_position"] and
            rsi < 75):
            signals.append({
                "idx": i,
                "type": "BUY",
                "rsi": rsi,
                "slope": slope,
                "adx": adx,
                "position": position
            })

        # SELL条件
        elif (slope < -params["slope_threshold"] and
              position > params["sell_position"] and
              rsi > 25):
            signals.append({
                "idx": i,
                "type": "SELL",
                "rsi": rsi,
                "slope": slope,
                "adx": adx,
                "position": position
            })

    return signals


def run_comparison(symbol: str = "XAUJPY", max_trades: int = 200, deposit: int = 50000):
    """閾値比較テスト実行"""

    print("=" * 70)
    print("[閾値比較テスト] ADX閾値による取引頻度・勝率への影響")
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
    print("=" * 70)

    lot_multiplier = deposit / 50000.0
    results_summary = []

    for pattern_name, params in THRESHOLD_PATTERNS.items():
        print(f"\n[{pattern_name}]")
        print(f"  ADX閾値: {params['adx_threshold']}")
        print(f"  Slope閾値: {params['slope_threshold']}")

        # シグナル検出
        signals = find_signals_with_params(prices, params)
        print(f"  検出シグナル数: {len(signals)}")

        if not signals:
            results_summary.append({
                "pattern": pattern_name,
                "signals": 0,
                "trades": 0,
                "wins": 0,
                "win_rate": 0,
                "pf": 0,
                "total_profit": 0
            })
            continue

        # ランダムサンプリング
        random.seed(42)
        test_signals = signals[:max_trades] if len(signals) <= max_trades else random.sample(signals, max_trades)

        # シミュレーション
        trade_results = []
        for signal in test_signals:
            result = simulate_trade(symbol, prices, signal["idx"], signal["type"], lot_multiplier)
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

        print(f"  取引数: {len(trade_results)}")
        print(f"  勝率: {win_rate:.1f}% ({len(wins)}/{len(trade_results)})")
        print(f"  PF: {pf:.2f}")
        print(f"  損益: {total_profit:+,.0f} JPY")

        results_summary.append({
            "pattern": pattern_name,
            "signals": len(signals),
            "trades": len(trade_results),
            "wins": len(wins),
            "win_rate": win_rate,
            "pf": pf,
            "total_profit": total_profit
        })

    # 比較サマリー
    print("\n" + "=" * 70)
    print("[比較サマリー]")
    print("=" * 70)
    print(f"{'パターン':<18} {'シグナル':<10} {'取引':<8} {'勝率':<10} {'PF':<8} {'損益':<14}")
    print("-" * 70)

    for r in results_summary:
        pf_str = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "∞"
        print(f"{r['pattern']:<18} {r['signals']:<10} {r['trades']:<8} {r['win_rate']:.1f}%{'':<5} {pf_str:<8} {r['total_profit']:+,.0f} JPY")

    print("=" * 70)

    # 推奨
    best = max(results_summary, key=lambda x: x['total_profit'] if x['pf'] > 0.8 else -999999)
    print(f"\n[推奨] {best['pattern']}")
    print(f"  理由: 利益 {best['total_profit']:+,.0f} JPY、勝率 {best['win_rate']:.1f}%、PF {best['pf']:.2f}")

    return results_summary


if __name__ == "__main__":
    run_comparison(symbol="XAUJPY", max_trades=200, deposit=50000)
