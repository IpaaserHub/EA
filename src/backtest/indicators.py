"""
Technical Indicators
====================
Pure Python implementations of trading indicators.
No external dependencies (pandas-ta doesn't support Python 3.14 yet).

These match the indicators in your main_genai_custom.py
"""

import statistics
from typing import List, Dict, Tuple


def calculate_rsi(prices: List[Dict], period: int = 14) -> float:
    """
    Calculate RSI (Relative Strength Index).

    RSI measures momentum - whether price is rising or falling.
    - RSI > 70: Overbought (price may fall)
    - RSI < 30: Oversold (price may rise)
    - RSI = 50: Neutral

    Args:
        prices: List of OHLC dicts [{"open": x, "high": x, "low": x, "close": x}, ...]
        period: Lookback period (default 14)

    Returns:
        RSI value (0-100)
    """
    if len(prices) < period + 1:
        return 50.0  # Neutral when not enough data

    closes = [p["close"] for p in prices]
    gains = []
    losses = []

    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    recent_gains = gains[-period:]
    recent_losses = losses[-period:]

    avg_gain = statistics.mean(recent_gains) if recent_gains else 0
    avg_loss = statistics.mean(recent_losses) if recent_losses else 0

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_adx(prices: List[Dict], period: int = 14) -> float:
    """
    Calculate ADX (Average Directional Index).

    ADX measures trend strength (not direction).
    - ADX < 20: No trend (range market)
    - ADX 20-25: Weak trend
    - ADX 25-50: Strong trend
    - ADX > 50: Very strong trend

    Args:
        prices: List of OHLC dicts
        period: Lookback period (default 14)

    Returns:
        ADX value (0-100)
    """
    if len(prices) < period + 1:
        return 20.0  # Neutral when not enough data

    closes = [p["close"] for p in prices]
    tr_list = []
    plus_dm_list = []
    minus_dm_list = []

    for i in range(1, len(closes)):
        move = closes[i] - closes[i - 1]
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

    # Simple smoothing
    tr_avg = statistics.mean(tr_list[-period:])
    plus_dm_avg = statistics.mean(plus_dm_list[-period:])
    minus_dm_avg = statistics.mean(minus_dm_list[-period:])

    if tr_avg == 0:
        return 20.0

    plus_di = 100 * plus_dm_avg / tr_avg
    minus_di = 100 * minus_dm_avg / tr_avg

    di_sum = plus_di + minus_di
    if di_sum == 0:
        return 20.0

    dx = 100 * abs(plus_di - minus_di) / di_sum
    return dx


def calculate_atr(prices: List[Dict], period: int = 14) -> float:
    """
    Calculate ATR (Average True Range).

    ATR measures volatility - how much price moves.
    Used for setting SL/TP distances.

    Args:
        prices: List of OHLC dicts
        period: Lookback period (default 14)

    Returns:
        ATR value
    """
    if len(prices) < period + 1:
        return 0.01  # Small default when not enough data

    closes = [p["close"] for p in prices]
    ranges = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    return statistics.mean(ranges[-period:])


def calculate_slope(prices: List[Dict]) -> float:
    """
    Calculate linear regression slope.

    Positive slope = uptrend
    Negative slope = downtrend
    Near zero = sideways

    Args:
        prices: List of OHLC dicts

    Returns:
        Slope value
    """
    closes = [p["close"] for p in prices]
    n = len(closes)
    if n < 2:
        return 0

    x = list(range(n))
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(closes)

    num = sum((x[i] - mean_x) * (closes[i] - mean_y) for i in range(n))
    den = sum((x[i] - mean_x) ** 2 for i in range(n))

    return num / den if den != 0 else 0


def calculate_position_in_range(prices: List[Dict], lookback: int = 20) -> float:
    """
    Calculate where current price sits in recent range.

    0.0 = at the low (good for buying)
    1.0 = at the high (good for selling)
    0.5 = middle of range

    Args:
        prices: List of OHLC dicts
        lookback: Number of candles for range calculation

    Returns:
        Position value (0.0 to 1.0)
    """
    if len(prices) < lookback:
        return 0.5

    recent = prices[-lookback:]
    current_price = prices[-1]["close"]

    high = max(p["high"] for p in recent)
    low = min(p["low"] for p in recent)

    if high == low:
        return 0.5

    return (current_price - low) / (high - low)


def calculate_all_indicators(prices: List[Dict], lookback: int = 50) -> Dict:
    """
    Calculate all indicators at once.

    Args:
        prices: List of OHLC dicts
        lookback: Window size for calculations

    Returns:
        Dict with all indicator values
    """
    window = prices[-lookback:] if len(prices) >= lookback else prices

    return {
        "rsi": calculate_rsi(window),
        "adx": calculate_adx(window),
        "atr": calculate_atr(window),
        "slope": calculate_slope(window),
        "position": calculate_position_in_range(window),
    }
