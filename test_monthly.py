"""
月別バックテスト
D1データを月ごとに分割してテスト
"""

import os
import random
from datetime import datetime
from test_historical import (
    SYMBOL_CONFIGS,
    calculate_rsi, calculate_slope, calculate_adx, calculate_position,
    simulate_trade
)

# H1用パラメータ（main_genai_custom.pyと同期）
ENTRY_PARAMS_H1 = {
    "XAUJPY": {
        "adx_threshold": 20,        # トレンド確認
        "slope_threshold": 0.00008, # H1用
        "buy_position": 0.45,       # 安値圏
        "sell_position": 0.55,      # 高値圏
        "rsi_buy_max": 65,
        "rsi_sell_min": 35,
        "rsi_extreme_avoid": False,
        "tp_mult": 3.0,             # ATR × 3.0
        "sl_mult": 2.0,             # ATR × 2.0
    },
    "DEFAULT": {
        "adx_threshold": 20,
        "slope_threshold": 0.00008,
        "buy_position": 0.45,
        "sell_position": 0.55,
        "rsi_buy_max": 65,
        "rsi_sell_min": 35,
        "rsi_extreme_avoid": False,
        "tp_mult": 3.0,
        "sl_mult": 2.0,
    }
}


def load_d1_data_with_dates(csv_path: str) -> list:
    """
    D1データを日付付きで読み込む
    Returns: [(date_str, close_price), ...]
    """
    if not os.path.exists(csv_path):
        print(f"  CSV not found: {csv_path}")
        return None

    data = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 最初の2行をスキップ（タイトル+ヘッダー）
        data_lines = lines[2:]

        for line in data_lines:
            row = line.strip().split(',')
            if len(row) >= 5:
                try:
                    date_str = row[0].strip()  # MM/DD/YYYY HH:MM
                    close_price = float(row[4].strip())
                    data.append((date_str, close_price))
                except (ValueError, IndexError):
                    continue

        # データは降順（新しい順）なので反転
        data.reverse()

        if data:
            print(f"  Loaded {len(data)} bars with dates")
            return data
    except Exception as e:
        print(f"  Error reading CSV: {e}")

    return None


def filter_by_month(data: list, year: int, month: int) -> list:
    """
    指定月のデータのみを抽出
    """
    filtered = []
    for date_str, price in data:
        try:
            # MM/DD/YYYY HH:MM format
            dt = datetime.strptime(date_str, "%m/%d/%Y %H:%M")
            if dt.year == year and dt.month == month:
                filtered.append((date_str, price))
        except:
            continue
    return filtered


def find_entry_signals_d1(prices: list, symbol: str, lookback: int = 20) -> list:
    """
    D1用エントリーシグナル検出
    日足は本数が少ないので短いルックバック
    """
    signals = []
    params = ENTRY_PARAMS_H1.get(symbol, ENTRY_PARAMS_H1["DEFAULT"])

    # 日足は最低10本あれば計算可能
    for i in range(lookback, len(prices) - 5):
        history = prices[max(0, i-lookback):i+1]

        # 各種指標を計算
        rsi = calculate_rsi(history, period=min(14, len(history)-1))
        slope = calculate_slope(history, period=min(20, len(history)-1))
        adx = calculate_adx(history, period=min(14, len(history)-1))
        position = calculate_position(history, period=min(20, len(history)-1))

        # ADXフィルター
        if adx < params["adx_threshold"]:
            continue

        # Slopeフィルター
        if abs(slope) < params["slope_threshold"]:
            continue

        # RSI極端値フィルター
        if params["rsi_extreme_avoid"] and (rsi < 25 or rsi > 75):
            continue

        # BUY条件
        if (slope > params["slope_threshold"] and
            position < params["buy_position"] and
            rsi < params["rsi_buy_max"]):
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
              rsi > params["rsi_sell_min"]):
            signals.append({
                "idx": i,
                "type": "SELL",
                "rsi": rsi,
                "slope": slope,
                "adx": adx,
                "position": position
            })

    return signals


def simulate_trade_d1(symbol: str, prices: list, entry_idx: int, trade_type: str) -> dict:
    """
    D1用トレードシミュレーション
    日足は値動きが大きいのでSL/TPを調整
    """
    params = ENTRY_PARAMS_H1.get(symbol, ENTRY_PARAMS_H1["DEFAULT"])
    config = SYMBOL_CONFIGS.get(symbol, SYMBOL_CONFIGS["USDJPY"])

    entry_price = prices[entry_idx]

    # ATR計算（日足用に短めの期間）
    lookback = min(14, entry_idx)
    if lookback < 5:
        # データ不足時は固定値
        atr = entry_price * 0.015  # 1.5%
    else:
        recent = prices[entry_idx - lookback:entry_idx + 1]
        diffs = [abs(recent[i] - recent[i-1]) for i in range(1, len(recent))]
        atr = sum(diffs) / len(diffs) if diffs else entry_price * 0.015

    # 日足用の大きめSL/TP
    tp_distance = atr * params["tp_mult"]
    sl_distance = atr * params["sl_mult"]

    if trade_type == "BUY":
        tp_price = entry_price + tp_distance
        sl_price = entry_price - sl_distance
    else:  # SELL
        tp_price = entry_price - tp_distance
        sl_price = entry_price + sl_distance

    # シミュレーション
    result = "TIMEOUT"
    exit_price = prices[-1] if entry_idx < len(prices) - 1 else entry_price
    bars_held = 0

    for j in range(entry_idx + 1, min(entry_idx + 30, len(prices))):  # 最大30日
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
        else:  # SELL
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

    profit = pips * config["lot_profit_per_pip"]

    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "result": result,
        "profit": profit,
        "pips": pips,
        "bars_held": bars_held
    }


def run_monthly_backtest(symbol: str = "XAUJPY", deposit: int = 50000):
    """
    月別バックテスト実行（H1データ使用）
    """
    # H1データファイルのパス（合成データ含む拡張版）
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    h1_path = os.path.join(data_dir, "XAUJPY-H1-extended-synthetic.csv")

    print("=" * 70)
    print("[月別バックテスト] XAUJPY H1（1時間足）")
    print("=" * 70)
    print(f"  入金額: {deposit:,} JPY")
    print(f"  ロット: 0.01 lot")
    print("=" * 70)

    # データ読み込み
    data = load_d1_data_with_dates(h1_path)
    if not data:
        print("データ読み込み失敗")
        return

    # 日付範囲を確認
    first_date = data[0][0]
    last_date = data[-1][0]
    print(f"  データ範囲: {first_date} ～ {last_date}")

    # 月別に分割（9月～1月）
    months = [
        (2025, 9, "2025年09月"),
        (2025, 10, "2025年10月"),
        (2025, 11, "2025年11月"),
        (2025, 12, "2025年12月"),
        (2026, 1, "2026年01月"),
    ]

    monthly_results = []
    all_prices = [price for _, price in data]  # 全期間の価格データ

    for year, month, label in months:
        month_data = filter_by_month(data, year, month)

        if len(month_data) < 100:  # H1は100本以上必要
            print(f"\n[{label}] データ不足 ({len(month_data)}本)")
            monthly_results.append({
                "month": label,
                "trades": 0,
                "wins": 0,
                "win_rate": 0,
                "pf": 0,
                "total_profit": 0,
                "roi": 0
            })
            continue

        # その月のデータのインデックス範囲を特定
        month_start_idx = None
        month_end_idx = None
        for i, (date_str, _) in enumerate(data):
            try:
                dt = datetime.strptime(date_str, "%m/%d/%Y %H:%M")
                if dt.year == year and dt.month == month:
                    if month_start_idx is None:
                        month_start_idx = i
                    month_end_idx = i
            except:
                continue

        if month_start_idx is None:
            print(f"\n[{label}] データなし")
            continue

        print(f"\n[{label}] {len(month_data)}本のデータ")

        # エントリーシグナル検出（全データを使用して指標計算）
        signals = []
        params = ENTRY_PARAMS_H1.get(symbol, ENTRY_PARAMS_H1["DEFAULT"])

        for i in range(month_start_idx, month_end_idx + 1):
            if i < 20:  # 最低20本のヒストリーが必要
                continue

            history = all_prices[max(0, i-20):i+1]

            rsi = calculate_rsi(history, period=min(14, len(history)-1))
            slope = calculate_slope(history, period=min(20, len(history)-1))
            adx = calculate_adx(history, period=min(14, len(history)-1))
            position = calculate_position(history, period=min(20, len(history)-1))

            if adx < params["adx_threshold"]:
                continue
            if abs(slope) < params["slope_threshold"]:
                continue

            # BUY条件
            if (slope > params["slope_threshold"] and
                position < params["buy_position"] and
                rsi < params["rsi_buy_max"]):
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
                  rsi > params["rsi_sell_min"]):
                signals.append({
                    "idx": i,
                    "type": "SELL",
                    "rsi": rsi,
                    "slope": slope,
                    "adx": adx,
                    "position": position
                })

        if not signals:
            print(f"  シグナルなし")
            monthly_results.append({
                "month": label,
                "trades": 0,
                "wins": 0,
                "win_rate": 0,
                "pf": 0,
                "total_profit": 0,
                "roi": 0
            })
            continue

        # シミュレーション実行
        results = []
        for signal in signals:
            result = simulate_trade_d1(symbol, all_prices, signal["idx"], signal["type"])
            result["symbol"] = symbol
            result["position_type"] = signal["type"]
            results.append(result)

        # 集計
        wins = [r for r in results if r["profit"] > 0]
        losses = [r for r in results if r["profit"] <= 0]
        total_profit = sum(r["profit"] for r in results)
        total_wins = sum(r["profit"] for r in wins) if wins else 0
        total_losses = abs(sum(r["profit"] for r in losses)) if losses else 0.01
        pf = total_wins / total_losses if total_losses > 0 else float('inf')
        win_rate = len(wins) / len(results) * 100 if results else 0
        roi = (total_profit / deposit) * 100

        monthly_results.append({
            "month": label,
            "trades": len(results),
            "wins": len(wins),
            "win_rate": win_rate,
            "pf": pf,
            "total_profit": total_profit,
            "roi": roi
        })

        print(f"  トレード数: {len(results)}")
        print(f"  勝率: {win_rate:.1f}% ({len(wins)}/{len(results)})")
        print(f"  PF: {pf:.2f}")
        print(f"  損益: {total_profit:+,.0f} JPY")
        print(f"  月利: {roi:+.1f}%")

    # サマリー
    print("\n" + "=" * 70)
    print("[月別サマリー]")
    print("=" * 70)
    print(f"{'月':<12} {'取引':<6} {'勝率':<8} {'PF':<8} {'損益':<12} {'月利':<8}")
    print("-" * 70)

    total_trades = 0
    total_wins = 0
    total_profit = 0

    for r in monthly_results:
        pf_str = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "∞"
        print(f"{r['month']:<12} {r['trades']:<6} {r['win_rate']:.1f}%{'':<3} {pf_str:<8} {r['total_profit']:+,.0f} JPY{'':<2} {r['roi']:+.1f}%")
        total_trades += r['trades']
        total_wins += r['wins']
        total_profit += r['total_profit']

    print("-" * 70)
    avg_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    avg_roi = (total_profit / deposit) * 100
    print(f"{'合計':<12} {total_trades:<6} {avg_win_rate:.1f}%{'':<3} {'-':<8} {total_profit:+,.0f} JPY{'':<2} {avg_roi:+.1f}%")
    print("=" * 70)

    return monthly_results


if __name__ == "__main__":
    random.seed(42)
    run_monthly_backtest(deposit=50000)
