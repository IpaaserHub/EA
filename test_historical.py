"""
Historical Data Backtest for AI Exit Decision
実際の過去市場データを使ったバックテスト
"""

import requests
import json
import time
import random
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
SERVER_URL = "http://127.0.0.1:8000"
ACCOUNT_ID = 75449373

# Symbol configurations with realistic parameters
SYMBOL_CONFIGS = {
    "USDJPY": {
        "pip_value": 0.01,  # 1 pip = 0.01
        "lot_profit_per_pip": 100,  # 0.01 lot = 100 yen per pip
        "typical_spread": 0.02,
    },
    "BTCJPY": {
        "pip_value": 1,
        "lot_profit_per_pip": 0.01,
        "typical_spread": 5000,
    },
    "XAUUSD": {
        "pip_value": 0.01,
        "lot_profit_per_pip": 100,
        "typical_spread": 0.30,
    },
    "XAUJPY": {
        "pip_value": 1,  # 1 pip = 1 JPY
        "lot_profit_per_pip": 1.0,  # 0.01 lot = 1円/pip (1oz × 1JPY = 1JPY)
        "typical_spread": 100,  # 約100円スプレッド
    }
}

# ============================================================
# v10.5: デュアルモード対応エントリーパラメータ
# STABLE: 安定重視（M15/H1）- 低頻度・高勝率
# AGGRESSIVE: 収益重視（M5）- 高頻度・アフィリエイト増
# ============================================================

# 現在のトレードモード（"STABLE" または "AGGRESSIVE"）
TRADE_MODE = "AGGRESSIVE"

# 安定モード: M15/H1用（日2-3回、高勝率）
ENTRY_PARAMS_STABLE = {
    "XAUJPY": {
        "adx_threshold": 15,
        "slope_threshold": 0.00004,
        "buy_position": 0.48,
        "sell_position": 0.52,
        "rsi_buy_max": 70,
        "rsi_sell_min": 30,
        "rsi_extreme_avoid": False,
        "tp_mult": 2.5,
        "sl_mult": 1.5,
    },
    "DEFAULT": {
        "adx_threshold": 20,
        "slope_threshold": 0.00008,
        "buy_position": 0.45,
        "sell_position": 0.55,
        "rsi_buy_max": 65,
        "rsi_sell_min": 35,
        "rsi_extreme_avoid": False,
        "tp_mult": 2.5,
        "sl_mult": 1.5,
    }
}

# 積極モード: M5用（日20回+、アフィリエイト重視）
ENTRY_PARAMS_AGGRESSIVE = {
    "XAUJPY": {
        "adx_threshold": 5,         # v10.6: 10→5に緩和（PF 1.12→1.41）
        "slope_threshold": 0.00001, # v10.6: 0.00002→0.00001に緩和
        "buy_position": 0.50,       # 緩和
        "sell_position": 0.50,      # 緩和
        "rsi_buy_max": 75,
        "rsi_sell_min": 25,
        "rsi_extreme_avoid": False,
        "tp_mult": 2.0,             # TP維持
        "sl_mult": 1.5,             # v10.7: 1.2→1.5（勝率48→53%、利益+8%）
    },
    "DEFAULT": {
        "adx_threshold": 10,
        "slope_threshold": 0.00002,
        "buy_position": 0.50,
        "sell_position": 0.50,
        "rsi_buy_max": 75,
        "rsi_sell_min": 25,
        "rsi_extreme_avoid": False,
        "tp_mult": 2.0,
        "sl_mult": 1.5,             # v10.7: 1.2→1.5
    }
}

def get_entry_params(mode: str = None):
    """トレードモードに応じたパラメータを取得"""
    if mode is None:
        mode = TRADE_MODE
    if mode == "AGGRESSIVE":
        return ENTRY_PARAMS_AGGRESSIVE
    return ENTRY_PARAMS_STABLE

# 後方互換性のためのエイリアス
ENTRY_PARAMS_V10 = ENTRY_PARAMS_STABLE

def load_new_csv_format(csv_path: str) -> list:
    """
    新形式CSVを読み込む（XAUJPY_historical_data25.9-26.1.05.csv用）
    フォーマット:
      行1: "XAUJPY Historical Data" (スキップ)
      行2: Date,Open,High,Low,Close,Change(Pips),Change(%) (ヘッダー)
      行3以降: MM/DD/YYYY HH:MM,Open,High,Low,Close,Change,Change%
    データは降順（新しい順）なので反転が必要
    """
    import os

    if not os.path.exists(csv_path):
        print(f"  CSV not found: {csv_path}")
        return None

    prices = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 最初の2行をスキップ（タイトル+ヘッダー）
        data_lines = lines[2:]

        for line in data_lines:
            row = line.strip().split(',')
            if len(row) >= 5:
                try:
                    # Close価格を取得（インデックス4）
                    close_str = row[4].strip()
                    close_price = float(close_str)
                    prices.append(close_price)
                except (ValueError, IndexError):
                    continue

        # データは降順（新しい順）なので反転
        prices.reverse()

        if prices:
            # ファイル名からタイムフレームを推定
            timeframe = "H1" if "1H" in csv_path or "1時間" in csv_path else "M15"
            print(f"  Loaded {len(prices)} bars from {os.path.basename(csv_path)} ({timeframe} data)")
            return prices
    except Exception as e:
        print(f"  Error reading new format CSV: {e}")

    return None


def load_mt5_csv(symbol: str, csv_path: str = None) -> list:
    """
    MT5からエクスポートしたCSVファイルを読み込む
    CSVフォーマット: Date,Time,Open,High,Low,Close,Volume
    """
    import os
    import csv

    data_dir = os.path.join(os.path.dirname(__file__), "data")

    # シンボル名マッピング（XAUUSD → XAUJPYなど）
    symbol_map = {
        "XAUUSD": "XAUJPY",  # ゴールドのJPY建て
    }
    mapped_symbol = symbol_map.get(symbol, symbol)

    # ファイル名パターンを試す（優先順）
    if csv_path is None:
        # M15データを優先（1日3回エントリー目標）
        m15_path = os.path.join(data_dir, "XAUJPY_historical_data25.9-26.1.05.csv")
        if os.path.exists(m15_path):
            csv_path = m15_path

        if csv_path is None:
            patterns = [
                # 新形式
                f"{mapped_symbol}_historical_data25.9-26.1.05.csv",
                f"{mapped_symbol}_H1_extended.csv",  # 拡張データ（UTF-8変換済み）
                f"{mapped_symbol}H1.csv",   # MT5エクスポート形式 (例: USDJPYH1.csv)
                f"{mapped_symbol}.csv",      # シンプル形式
                f"{symbol}H1.csv",           # オリジナルシンボル名
                f"{symbol}.csv",
            ]

            for pattern in patterns:
                test_path = os.path.join(data_dir, pattern)
                if os.path.exists(test_path):
                    csv_path = test_path
                    break

        if csv_path is None:
            print(f"  CSV not found for {symbol} (tried: {patterns})")
            return None

    if not os.path.exists(csv_path):
        print(f"  CSV not found: {csv_path}")
        return None

    # 新形式CSVかチェック
    if "historical_data" in csv_path:
        return load_new_csv_format(csv_path)

    prices = []
    try:
        # MT5 CSVはUTF-16またはUTF-8、カンマまたはタブ区切り
        encodings = ['utf-16', 'utf-8', 'utf-8-sig', 'cp932']
        delimiters = [',', '\t']

        content = None
        for enc in encodings:
            try:
                with open(csv_path, 'r', encoding=enc) as f:
                    content = f.read()
                    break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if content is None:
            print(f"  Failed to decode {csv_path}")
            return None

        lines = content.strip().split('\n')

        # 区切り文字を判定
        delimiter = ','
        if lines and '\t' in lines[0] and ',' not in lines[0]:
            delimiter = '\t'

        for line in lines:
            row = line.strip().split(delimiter)
            if len(row) >= 5:
                try:
                    # Close価格を取得（インデックス4 = 5列目）
                    # フォーマット: DateTime,Open,High,Low,Close,Volume,...
                    close_price = float(row[4])
                    prices.append(close_price)
                except (ValueError, IndexError):
                    continue

        if prices:
            print(f"  Loaded {len(prices)} bars from {os.path.basename(csv_path)}")
            return prices
    except Exception as e:
        print(f"  Error reading CSV: {e}")

    return None


def fetch_historical_data(symbol: str, period: str = "3mo", interval: str = "1h"):
    """
    履歴データを取得（優先順位: MT5 CSV → yfinance → 合成データ）
    """
    # 1. MT5 CSVファイルをチェック
    csv_data = load_mt5_csv(symbol)
    if csv_data and len(csv_data) >= 200:
        return csv_data

    # 2. yfinance を試す
    try:
        import yfinance as yf

        ticker_map = {
            "USDJPY": "USDJPY=X",
            "BTCJPY": "BTC-JPY",
            "XAUUSD": "GC=F",  # Gold Futures
        }

        ticker = ticker_map.get(symbol, f"{symbol}=X")
        data = yf.download(ticker, period=period, interval=interval, progress=False)

        if not data.empty:
            prices = data['Close'].tolist()
            print(f"  Fetched {len(prices)} bars from yfinance")
            return prices

    except ImportError:
        print(f"  yfinance not installed")
    except Exception as e:
        print(f"  yfinance error: {e}")

    # 3. 合成データにフォールバック
    print(f"  Using synthetic data for {symbol}")
    return generate_synthetic_data(symbol, 500)


def fetch_historical_data_yfinance(symbol: str, period: str = "3mo", interval: str = "1h"):
    """
    Yahoo Finance APIから過去データを取得（後方互換性のため維持）
    """
    return fetch_historical_data(symbol, period, interval)

def generate_synthetic_data(symbol: str, num_bars: int) -> list:
    """
    実際の相場パターンを模したシンセティックデータ生成
    - トレンド期間とレンジ期間を含む
    - 現実的なボラティリティ
    """
    base_prices = {
        "USDJPY": 157.50,
        "BTCJPY": 14000000,
        "XAUUSD": 2650.0
    }

    volatilities = {
        "USDJPY": 0.0003,  # 0.03%
        "BTCJPY": 0.005,   # 0.5%
        "XAUUSD": 0.003,   # 0.3%
    }

    base = base_prices.get(symbol, 100)
    vol = volatilities.get(symbol, 0.001)

    prices = [base]
    trend = 0  # -1: down, 0: range, 1: up
    trend_duration = 0

    for i in range(num_bars - 1):
        # Change trend randomly
        if trend_duration <= 0:
            trend = random.choice([-1, 0, 0, 1])  # Range is more common
            trend_duration = random.randint(10, 50)

        trend_duration -= 1

        # Price movement
        drift = trend * vol * 0.3  # Trend component
        noise = random.gauss(0, vol)  # Random component

        new_price = prices[-1] * (1 + drift + noise)
        prices.append(new_price)

    return prices

def calculate_rsi(prices: list, period: int = 14) -> float:
    """RSI計算"""
    if len(prices) < period + 1:
        return 50.0

    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_trend(prices: list, period: int = 20) -> str:
    """線形回帰でトレンド判定"""
    if len(prices) < period:
        return "range"

    recent = prices[-period:]
    n = len(recent)
    x_mean = (n - 1) / 2
    y_mean = sum(recent) / n

    numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return "range"

    slope = numerator / denominator

    # Normalize slope by price level
    normalized_slope = slope / y_mean

    if normalized_slope > 0.0001:
        return "up"
    elif normalized_slope < -0.0001:
        return "down"
    else:
        return "range"

def calculate_slope(prices: list, period: int = 20) -> float:
    """線形回帰でslope値を計算（v10.0用）"""
    if len(prices) < period:
        return 0.0

    recent = prices[-period:]
    n = len(recent)
    x_mean = (n - 1) / 2
    y_mean = sum(recent) / n

    numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return 0.0

    slope = numerator / denominator
    # Normalize by price level
    return slope / y_mean if y_mean != 0 else 0.0

def calculate_adx(prices: list, period: int = 14) -> float:
    """
    ADX（平均方向性指数）を計算（v10.0用）
    簡略化版: 終値のみから計算
    """
    if len(prices) < period + 1:
        return 20.0

    tr_list = []
    plus_dm_list = []
    minus_dm_list = []

    for i in range(1, len(prices)):
        move = prices[i] - prices[i-1]
        tr = abs(move)
        tr_list.append(tr if tr > 0 else 0.0001)

        if move > 0:
            plus_dm_list.append(move)
            minus_dm_list.append(0)
        else:
            plus_dm_list.append(0)
            minus_dm_list.append(abs(move))

    if len(tr_list) < period:
        return 20.0

    def wilders_smooth(data, p):
        smoothed = [sum(data[:p]) / p]
        for i in range(p, len(data)):
            smoothed.append((smoothed[-1] * (p - 1) + data[i]) / p)
        return smoothed

    tr_smooth = wilders_smooth(tr_list, period)
    plus_dm_smooth = wilders_smooth(plus_dm_list, period)
    minus_dm_smooth = wilders_smooth(minus_dm_list, period)

    if not tr_smooth:
        return 20.0

    dx_list = []
    for i in range(len(tr_smooth)):
        if tr_smooth[i] > 0:
            plus_di = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di = 100 * minus_dm_smooth[i] / tr_smooth[i]

            di_sum = plus_di + minus_di
            if di_sum > 0:
                dx = 100 * abs(plus_di - minus_di) / di_sum
                dx_list.append(dx)

    if not dx_list:
        return 20.0

    return sum(dx_list[-period:]) / min(period, len(dx_list))

def calculate_position(prices: list, period: int = 100) -> float:
    """現在価格の相対位置を計算（v10.0用）"""
    if len(prices) < period:
        return 0.5

    recent = prices[-period:]
    highest = max(recent)
    lowest = min(recent)
    current = prices[-1]

    price_range = highest - lowest
    if price_range <= 0:
        return 0.5

    return (current - lowest) / price_range

def simulate_trade(symbol: str, prices: list, entry_idx: int, position_type: str, lot_multiplier: float = 1.0) -> dict:
    """
    単一トレードのシミュレーション v9.0
    - トレーリングストップ
    - ブレークイーブン
    - 分割決済
    - lot_multiplier: 入金額に応じたロット倍率（基準: 50,000円 = 1.0）
    """
    config = SYMBOL_CONFIGS.get(symbol, SYMBOL_CONFIGS["USDJPY"])

    entry_price = prices[entry_idx]

    # SL/TP設定（ATRベース、シンボル別調整）
    recent_prices = prices[max(0, entry_idx-20):entry_idx+1]
    atr = sum(abs(recent_prices[i] - recent_prices[i-1]) for i in range(1, len(recent_prices))) / len(recent_prices)

    # v10.0: 高勝率型SL/TP（TP狭め、SL広め）
    params = ENTRY_PARAMS_V10.get(symbol, ENTRY_PARAMS_V10["DEFAULT"])
    cfg = {"sl_mult": params["sl_mult"], "tp_mult": params["tp_mult"]}

    if position_type == "BUY":
        sl = entry_price - atr * cfg["sl_mult"]
        tp = entry_price + atr * cfg["tp_mult"]
    else:
        sl = entry_price + atr * cfg["sl_mult"]
        tp = entry_price - atr * cfg["tp_mult"]

    # v9.0: 状態変数
    original_sl = sl
    max_profit_seen = 0.0
    partial_closed = False
    partial_profit_locked = 0.0  # 分割決済で確定した利益
    remaining_ratio = 1.0  # 残りポジション比率

    # 価格を進めながらAI判断をシミュレート
    max_bars = min(100, len(prices) - entry_idx - 1)

    for bars_held in range(1, max_bars):
        current_idx = entry_idx + bars_held
        current_price = prices[current_idx]

        # 含み損益計算
        if position_type == "BUY":
            profit_pips = (current_price - entry_price) / config["pip_value"]
        else:
            profit_pips = (entry_price - current_price) / config["pip_value"]

        profit_yen = profit_pips * config["lot_profit_per_pip"] * lot_multiplier * remaining_ratio

        # v9.0: 最高利益を更新
        if profit_yen > max_profit_seen:
            max_profit_seen = profit_yen

        # SL/TPヒットチェック（v9.0: 動的SLに対応）
        if position_type == "BUY":
            if current_price <= sl:
                # SL到達時、利益がゼロ以上なら「ブレークイーブン成功」
                final_profit = partial_profit_locked + profit_yen
                reason = "BE_HIT" if sl > original_sl else "SL_HIT"
                return {
                    "exit_reason": reason,
                    "profit": final_profit,
                    "bars_held": bars_held,
                    "partial_closed": partial_closed,
                    "max_profit_seen": max_profit_seen
                }
            if current_price >= tp:
                return {
                    "exit_reason": "TP_HIT",
                    "profit": partial_profit_locked + abs(profit_yen),
                    "bars_held": bars_held,
                    "partial_closed": partial_closed,
                    "max_profit_seen": max_profit_seen
                }
        else:
            if current_price >= sl:
                final_profit = partial_profit_locked + profit_yen
                reason = "BE_HIT" if sl < original_sl else "SL_HIT"
                return {
                    "exit_reason": reason,
                    "profit": final_profit,
                    "bars_held": bars_held,
                    "partial_closed": partial_closed,
                    "max_profit_seen": max_profit_seen
                }
            if current_price <= tp:
                return {
                    "exit_reason": "TP_HIT",
                    "profit": partial_profit_locked + abs(profit_yen),
                    "bars_held": bars_held,
                    "partial_closed": partial_closed,
                    "max_profit_seen": max_profit_seen
                }

        # 5バーごとにAI判断を呼び出し（60秒間隔をシミュレート）
        if bars_held % 5 == 0:
            # RSIとトレンドを計算
            history = prices[max(0, current_idx-50):current_idx+1]
            rsi = calculate_rsi(history)
            trend = calculate_trend(history)

            # AI決済判断API呼び出し（v9.0対応）
            try:
                response = requests.post(
                    f"{SERVER_URL}/check_exit",
                    json={
                        "account_id": ACCOUNT_ID,
                        "ticket": random.randint(10000, 99999),
                        "symbol": symbol,
                        "position_type": position_type,
                        "open_price": entry_price,
                        "current_price": current_price,
                        "profit": profit_yen,
                        "volume": 0.01 * remaining_ratio,
                        "open_time": int(time.time()) - bars_held * 60,
                        "sl": sl,
                        "tp": tp,
                        "prices": history,
                        # v9.0追加フィールド
                        "max_profit_seen": max_profit_seen,
                        "partial_closed": partial_closed
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    action = result.get("action")

                    # v9.0: アクション別処理
                    if action == "CLOSE":
                        return {
                            "exit_reason": f"AI_{result.get('reason', 'Unknown')}",
                            "profit": partial_profit_locked + profit_yen,
                            "bars_held": bars_held,
                            "partial_closed": partial_closed,
                            "max_profit_seen": max_profit_seen
                        }

                    elif action == "PARTIAL_CLOSE" and not partial_closed:
                        # 分割決済：利益の一部を確定
                        close_ratio = result.get("partial_ratio", 0.5)
                        partial_profit_locked = profit_yen * close_ratio
                        remaining_ratio = 1.0 - close_ratio
                        partial_closed = True
                        # SLをブレークイーブンに移動
                        if position_type == "BUY":
                            sl = max(sl, entry_price + 10)  # 建値+バッファ
                        else:
                            sl = min(sl, entry_price - 10)

                    elif action == "MODIFY_SL":
                        # トレーリングストップまたはブレークイーブン
                        new_sl = result.get("new_sl", 0.0)
                        if new_sl > 0:
                            if position_type == "BUY" and new_sl > sl:
                                sl = new_sl
                            elif position_type == "SELL" and new_sl < sl:
                                sl = new_sl

            except Exception as e:
                pass  # Continue if API fails

    # タイムアウト（最大保有時間到達）
    final_price = prices[entry_idx + max_bars]
    if position_type == "BUY":
        final_profit = ((final_price - entry_price) / config["pip_value"]) * config["lot_profit_per_pip"] * lot_multiplier * remaining_ratio
    else:
        final_profit = ((entry_price - final_price) / config["pip_value"]) * config["lot_profit_per_pip"] * lot_multiplier * remaining_ratio

    return {
        "exit_reason": "TIMEOUT",
        "profit": partial_profit_locked + final_profit,
        "bars_held": max_bars,
        "partial_closed": partial_closed,
        "max_profit_seen": max_profit_seen
    }

def find_entry_signals(prices: list, symbol: str, mode: str = None) -> list:
    """
    v10.5: デュアルモード対応エントリーシグナル検出

    Args:
        prices: 価格データ
        symbol: シンボル名
        mode: "STABLE"（安定）または "AGGRESSIVE"（積極）
    """
    signals = []
    entry_params = get_entry_params(mode)
    params = entry_params.get(symbol, entry_params["DEFAULT"])

    for i in range(100, len(prices) - 100):  # 前後に余裕を持つ
        history = prices[i-100:i+1]

        # v10.0: 各種指標を計算
        rsi = calculate_rsi(history)
        slope = calculate_slope(history, period=50)
        adx = calculate_adx(history)
        position = calculate_position(history)

        # --- v10.0: ADXフィルター ---
        if adx < params["adx_threshold"]:
            continue  # 弱トレンド、スキップ

        # --- v10.0: Slopeフィルター ---
        if abs(slope) < params["slope_threshold"]:
            continue  # トレンド不明確、スキップ

        # --- v10.0: RSI極端値フィルター ---
        if params["rsi_extreme_avoid"] and (rsi < 25 or rsi > 75):
            continue  # 極端値、スキップ

        # --- v10.0: BUY条件 ---
        # 強い上昇トレンド + 深い安値圏 + RSI過熱なし
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

        # --- v10.0: SELL条件 ---
        # 強い下降トレンド + 深い高値圏 + RSI過熱なし
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

def analyze_ai_accuracy(results: list) -> dict:
    """
    AI判断の正確性を分析（v9.0対応）
    - FalseClose: AIがCLOSEしたが、HOLDすればより良い結果だった
    - FalseHold: AIがHOLDしたが、CLOSEすべきだった（SL到達など）
    - v9.0: トレーリング/BE/分割決済の効果
    """
    ai_decisions = [r for r in results if r["exit_reason"].startswith("AI_")]
    non_ai_exits = [r for r in results if not r["exit_reason"].startswith("AI_")]

    # AI判断の分類
    ai_close = len(ai_decisions)  # AIがCLOSE判断した数
    ai_hold_implied = len([r for r in results if r["exit_reason"] in ["SL_HIT", "TP_HIT", "TIMEOUT", "BE_HIT"]])

    # FalseClose: AIがCLOSEしたが損失だった（利益が伸びた可能性）
    false_close = len([r for r in ai_decisions if r["profit"] < 0])

    # FalseHold: SLに到達した（AIがHOLDし続けた結果）
    false_hold = len([r for r in results if r["exit_reason"] == "SL_HIT"])

    # v9.0: ブレークイーブン成功数
    be_hits = len([r for r in results if r["exit_reason"] == "BE_HIT"])

    # v9.0: 分割決済が発動したトレード数
    partial_closed_count = len([r for r in results if r.get("partial_closed", False)])

    # 正解判定
    # - AIがCLOSEして利益 → 正解
    # - TPに到達 → 正解（HOLDして利益伸ばした）
    # - v9.0: BE_HITも正解（損失回避成功）
    ai_correct_close = len([r for r in ai_decisions if r["profit"] > 0])
    tp_hits = len([r for r in results if r["exit_reason"] == "TP_HIT"])

    total_decisions = len(results)
    correct_decisions = ai_correct_close + tp_hits + be_hits
    accuracy = (correct_decisions / total_decisions * 100) if total_decisions > 0 else 0

    return {
        "total": total_decisions,
        "ai_close": ai_close,
        "ai_hold_implied": ai_hold_implied,
        "false_close": false_close,
        "false_hold": false_hold,
        "be_hits": be_hits,
        "partial_closed": partial_closed_count,
        "accuracy": accuracy,
        "correct": correct_decisions
    }


def print_symbol_report(symbol: str, results: list, deposit: int = 1000000):
    """ペア別詳細レポート（v9.0対応）"""
    if not results:
        print(f"\n{'='*70}")
        print(f"[{symbol}] No results")
        return

    wins = [r for r in results if r["profit"] > 0]
    losses = [r for r in results if r["profit"] <= 0]
    total_profit = sum(r["profit"] for r in results)
    total_wins = sum(r["profit"] for r in wins) if wins else 0
    total_losses = abs(sum(r["profit"] for r in losses)) if losses else 0.01
    pf = total_wins / total_losses if total_losses > 0 else float('inf')

    ai_stats = analyze_ai_accuracy(results)

    # Exit reason breakdown
    exit_reasons = {}
    for r in results:
        reason = r["exit_reason"].split("_")[0] if "_" in r["exit_reason"] else r["exit_reason"]
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    print(f"\n{'='*70}")
    print(f"[{symbol}] 詳細レポート v9.0")
    print(f"{'='*70}")
    print(f"  入金想定額      : {deposit:,} JPY")
    print(f"  テスト数        : {len(results)}")
    print(f"  ")
    print(f"  --- AI判断分析 ---")
    print(f"  AI正解率        : {ai_stats['accuracy']:.1f}% ({ai_stats['correct']}/{ai_stats['total']})")
    print(f"  AI判断分布      : CLOSE={ai_stats['ai_close']}, HOLD(暗黙)={ai_stats['ai_hold_implied']}")
    print(f"  ")
    print(f"  --- エラー分析 ---")
    print(f"  FalseClose      : {ai_stats['false_close']} (AIがCLOSE→損失)")
    print(f"  FalseHold       : {ai_stats['false_hold']} (HOLD継続→SL到達)")
    print(f"  ")
    print(f"  --- v9.0新機能効果 ---")
    print(f"  ブレークイーブン: {ai_stats['be_hits']} 回発動（損失回避）")
    print(f"  分割決済       : {ai_stats['partial_closed']} 回発動（利益確定）")
    print(f"  ")
    print(f"  --- 収益分析 ---")
    print(f"  勝率            : {len(wins)}/{len(results)} ({100*len(wins)/len(results):.1f}%)")
    print(f"  平均勝ち        : {total_wins/len(wins):+,.0f} JPY" if wins else "  平均勝ち        : N/A")
    print(f"  平均負け        : {-total_losses/len(losses):,.0f} JPY" if losses else "  平均負け        : N/A")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  総損益          : {total_profit:+,.0f} JPY")
    print(f"  ")
    print(f"  --- 決済理由 ---")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")


def print_overall_report(all_results: list, symbols: list, deposit: int = 1000000):
    """総合レポート（v9.0対応）"""
    if not all_results:
        print("\n[OVERALL] No results")
        return

    wins = [r for r in all_results if r["profit"] > 0]
    losses = [r for r in all_results if r["profit"] <= 0]
    total_profit = sum(r["profit"] for r in all_results)
    total_wins = sum(r["profit"] for r in wins) if wins else 0
    total_losses = abs(sum(r["profit"] for r in losses)) if losses else 0.01
    pf = total_wins / total_losses if total_losses > 0 else float('inf')

    ai_stats = analyze_ai_accuracy(all_results)

    # 予測計算
    trades_per_day = 5  # 1日あたりの取引数（推定）
    trades_per_month = trades_per_day * 22  # 月22営業日
    avg_profit_per_trade = total_profit / len(all_results)
    daily_pl = avg_profit_per_trade * trades_per_day
    monthly_pl = avg_profit_per_trade * trades_per_month
    monthly_roi = (monthly_pl / deposit) * 100

    print(f"\n{'='*70}")
    print(f"[総合結果 v9.0] {', '.join(symbols)}")
    print(f"{'='*70}")
    print(f"  入金想定額      : {deposit:,} JPY")
    print(f"  総テスト数      : {len(all_results)}")
    print(f"  ")
    print(f"  --- AI判断分析 ---")
    print(f"  総AI正解率      : {ai_stats['accuracy']:.1f}%")
    print(f"  AI出力比率      : CLOSE {ai_stats['ai_close']/len(all_results)*100:.1f}% / HOLD {ai_stats['ai_hold_implied']/len(all_results)*100:.1f}%")
    print(f"  総エラー数      : {ai_stats['false_close'] + ai_stats['false_hold']} (FC:{ai_stats['false_close']} FH:{ai_stats['false_hold']})")
    print(f"  ")
    print(f"  --- v9.0新機能効果 ---")
    print(f"  ブレークイーブン: {ai_stats['be_hits']} 回発動（損失回避成功）")
    print(f"  分割決済       : {ai_stats['partial_closed']} 回発動（利益確定）")
    print(f"  FalseHold削減  : {ai_stats['false_hold']}/{len(all_results)} ({100*ai_stats['false_hold']/len(all_results):.1f}%)")
    print(f"  ")
    print(f"  --- 収益分析 ---")
    print(f"  勝率            : {len(wins)}/{len(all_results)} ({100*len(wins)/len(all_results):.1f}%)")
    print(f"  平均勝ち        : {total_wins/len(wins):+,.0f} JPY" if wins else "  平均勝ち        : N/A")
    print(f"  平均負け        : {-total_losses/len(losses):,.0f} JPY" if losses else "  平均負け        : N/A")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  総損益          : {total_profit:+,.0f} JPY")
    print(f"  ")
    print(f"  --- 予測 ---")
    print(f"  日次予測損益    : {daily_pl:+,.0f} JPY")
    print(f"  月次予測損益    : {monthly_pl:+,.0f} JPY")
    print(f"  月次ROI         : {monthly_roi:+.2f}%")
    print(f"{'='*70}")


def run_historical_backtest(symbols: list = None, max_trades: int = 50, deposit: int = 1000000, mode: str = None):
    """
    ヒストリカルバックテスト実行（ルールブック準拠版）

    Args:
        symbols: テスト対象シンボル
        max_trades: 最大取引数
        deposit: 入金額（JPY）
        mode: "STABLE"（安定）または "AGGRESSIVE"（積極）
    """
    if symbols is None:
        symbols = ["USDJPY", "BTCJPY", "XAUUSD"]

    # モード判定
    if mode is None:
        mode = TRADE_MODE
    mode_label = "安定" if mode == "STABLE" else "積極"

    # ロット倍率計算（基準: 50,000円 = 0.01lot = 1.0倍）
    lot_multiplier = deposit / 50000.0
    calculated_lot = 0.01 * lot_multiplier

    print("=" * 70)
    print(f"[HISTORICAL BACKTEST] v10.5 デュアルモード対応")
    print("=" * 70)
    print(f"  Date/Time    : {datetime.now()}")
    print(f"  Trade Mode   : {mode} ({mode_label})")
    print(f"  Server       : {SERVER_URL}")
    print(f"  Symbols      : {', '.join(symbols)}")
    print(f"  Max Trades   : {max_trades} per symbol")
    print(f"  Deposit      : {deposit:,} JPY")
    print(f"  Lot Size     : {calculated_lot:.2f} lot ({lot_multiplier:.1f}x)")
    print("=" * 70)

    all_results = []
    symbol_results_map = {}

    for symbol in symbols:
        print(f"\n[{symbol}] Fetching historical data...")
        prices = fetch_historical_data_yfinance(symbol)

        if len(prices) < 200:
            print(f"  Insufficient data for {symbol}, skipping")
            continue

        # Send price history to server
        try:
            requests.post(
                f"{SERVER_URL}/history",
                json={"account_id": ACCOUNT_ID, "symbol": symbol, "prices": prices[-100:]},
                timeout=5
            )
        except:
            pass

        # Find entry signals (モード対応)
        signals = find_entry_signals(prices, symbol, mode)
        print(f"  Found {len(signals)} potential entry signals")

        # Limit trades
        if len(signals) > max_trades:
            signals = random.sample(signals, max_trades)

        # Simulate each trade
        print(f"  Simulating {len(signals)} trades...")
        symbol_results = []

        for i, signal in enumerate(signals):
            result = simulate_trade(symbol, prices, signal["idx"], signal["type"], lot_multiplier)
            result["symbol"] = symbol
            result["position_type"] = signal["type"]
            result["entry_rsi"] = signal["rsi"]
            # v10.0: trendキーがない場合はslopeから生成
            if "trend" in signal:
                result["entry_trend"] = signal["trend"]
            else:
                slope = signal.get("slope", 0)
                result["entry_trend"] = f"v10_{'Up' if slope > 0 else 'Down'}_ADX{signal.get('adx', 0):.0f}"
            symbol_results.append(result)

            if (i + 1) % 10 == 0:
                print(f"    Progress: {i+1}/{len(signals)}")

        symbol_results_map[symbol] = symbol_results
        all_results.extend(symbol_results)

    # ペア別詳細レポート
    for symbol in symbols:
        if symbol in symbol_results_map:
            print_symbol_report(symbol, symbol_results_map[symbol], deposit)

    # 総合レポート
    print_overall_report(all_results, symbols, deposit)

    return all_results

if __name__ == "__main__":
    # v8.0: シード固定で再現性確保
    random.seed(42)

    # Run backtest
    # v10.1: 月利10～50%目標（入金額5～10万円）
    results = run_historical_backtest(
        symbols=["XAUJPY"],  # ゴールド専用
        max_trades=500,  # Per symbol
        deposit=50000  # 入金5万円（推奨設定ガイドLv.1）
    )
