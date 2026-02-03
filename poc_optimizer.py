"""
Proof of Concept: Self-Improving Parameter Optimizer
=====================================================
This is a SIMPLE demo to show how Optuna finds better trading parameters.

Run with: python poc_optimizer.py

What this does:
1. Loads your historical price data
2. Uses Optuna to try different parameter combinations
3. Runs a simple backtest for each combination
4. Finds the best parameters automatically

This is NOT the full system - just a demo to understand the concept.
"""

import statistics
import os

# Check if optuna is installed
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("Optuna not installed. Installing now...")
    os.system("pip install optuna")
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# STEP 1: Load Price Data
# ============================================================

def load_price_data(csv_path: str) -> list:
    """
    Load OHLC data from CSV file.
    Returns list of dicts: [{"open": x, "high": x, "low": x, "close": x}, ...]
    """
    prices = []

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            # Skip header lines
            if 'Date' in line or 'Historical' in line:
                continue

            row = line.strip().split(',')
            if len(row) >= 5:
                try:
                    prices.append({
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4])
                    })
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return []

    print(f"Loaded {len(prices)} candles from {csv_path}")
    return prices


# ============================================================
# STEP 2: Technical Indicators (same as your main_genai_custom.py)
# ============================================================

def calculate_rsi(prices: list, period: int = 14) -> float:
    """Calculate RSI (Relative Strength Index)"""
    if len(prices) < period + 1:
        return 50.0

    closes = [p["close"] for p in prices]
    gains = []
    losses = []

    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
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


def calculate_adx(prices: list, period: int = 14) -> float:
    """Calculate ADX (Average Directional Index) - simplified version"""
    if len(prices) < period + 1:
        return 20.0

    closes = [p["close"] for p in prices]
    tr_list = []
    plus_dm_list = []
    minus_dm_list = []

    for i in range(1, len(closes)):
        move = closes[i] - closes[i-1]
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


def calculate_atr(prices: list, period: int = 14) -> float:
    """Calculate ATR (Average True Range)"""
    if len(prices) < period + 1:
        return 0.01

    closes = [p["close"] for p in prices]
    ranges = [abs(closes[i] - closes[i-1]) for i in range(1, len(closes))]
    return statistics.mean(ranges[-period:])


def linear_regression_slope(prices: list) -> float:
    """Calculate linear regression slope"""
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


# ============================================================
# STEP 3: Simple Backtest (same logic as your system)
# ============================================================

def run_backtest(prices: list, params: dict) -> dict:
    """
    Run a simple backtest with given parameters.

    This simulates your trading logic:
    - Check entry conditions (ADX, slope, RSI, position in range)
    - Calculate SL/TP based on ATR
    - Track wins/losses

    Returns metrics: win_rate, profit_factor, total_profit, num_trades
    """
    if len(prices) < 100:
        return {"win_rate": 0, "profit_factor": 0, "total_profit": 0, "num_trades": 0}

    # Parameters to test
    adx_threshold = params["adx_threshold"]
    slope_threshold = params["slope_threshold"]
    tp_mult = params["tp_mult"]
    sl_mult = params["sl_mult"]
    rsi_buy_max = params.get("rsi_buy_max", 75)
    rsi_sell_min = params.get("rsi_sell_min", 25)

    wins = 0
    losses = 0
    total_profit = 0
    gross_profit = 0
    gross_loss = 0

    # Simulate trading
    lookback = 50  # Candles to analyze

    for i in range(lookback, len(prices) - 20):  # Leave room for trade to play out
        window = prices[i-lookback:i]
        current_price = prices[i]["close"]

        # Calculate indicators
        adx = calculate_adx(window)
        rsi = calculate_rsi(window)
        slope = linear_regression_slope(window)
        atr = calculate_atr(window)

        # Calculate position in range
        high = max(p["high"] for p in window[-20:])
        low = min(p["low"] for p in window[-20:])
        position = (current_price - low) / (high - low) if high != low else 0.5

        signal = None

        # Check BUY conditions (same logic as your system)
        if (adx > adx_threshold and
            slope > slope_threshold and
            position < 0.5 and
            rsi < rsi_buy_max):
            signal = "BUY"

        # Check SELL conditions
        elif (adx > adx_threshold and
              slope < -slope_threshold and
              position > 0.5 and
              rsi > rsi_sell_min):
            signal = "SELL"

        if signal:
            # Calculate SL and TP
            if signal == "BUY":
                sl = current_price - (atr * sl_mult)
                tp = current_price + (atr * tp_mult)
            else:
                sl = current_price + (atr * sl_mult)
                tp = current_price - (atr * tp_mult)

            # Simulate trade outcome (look at next 20 candles)
            for j in range(i + 1, min(i + 21, len(prices))):
                future_high = prices[j]["high"]
                future_low = prices[j]["low"]

                if signal == "BUY":
                    # Check if TP hit
                    if future_high >= tp:
                        profit = tp - current_price
                        wins += 1
                        total_profit += profit
                        gross_profit += profit
                        break
                    # Check if SL hit
                    elif future_low <= sl:
                        loss = current_price - sl
                        losses += 1
                        total_profit -= loss
                        gross_loss += loss
                        break
                else:  # SELL
                    if future_low <= tp:
                        profit = current_price - tp
                        wins += 1
                        total_profit += profit
                        gross_profit += profit
                        break
                    elif future_high >= sl:
                        loss = sl - current_price
                        losses += 1
                        total_profit -= loss
                        gross_loss += loss
                        break

    num_trades = wins + losses
    win_rate = (wins / num_trades * 100) if num_trades > 0 else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_profit": total_profit,
        "num_trades": num_trades,
        "wins": wins,
        "losses": losses
    }


# ============================================================
# STEP 4: Optuna Optimization
# ============================================================

def create_objective(prices: list):
    """
    Create an Optuna objective function.

    Optuna will call this function many times with different parameters,
    trying to find the combination that gives the highest score.
    """
    def objective(trial):
        # Optuna suggests parameters within these ranges
        params = {
            "adx_threshold": trial.suggest_int("adx_threshold", 3, 30),
            "slope_threshold": trial.suggest_float("slope_threshold", 0.000001, 0.0001, log=True),
            "tp_mult": trial.suggest_float("tp_mult", 1.0, 4.0),
            "sl_mult": trial.suggest_float("sl_mult", 0.8, 3.0),
            "rsi_buy_max": trial.suggest_int("rsi_buy_max", 60, 80),
            "rsi_sell_min": trial.suggest_int("rsi_sell_min", 20, 40),
        }

        # Run backtest with these parameters
        results = run_backtest(prices, params)

        # Score: we want high profit factor AND reasonable number of trades
        # Penalize if too few trades (might be overfitting)
        if results["num_trades"] < 10:
            return 0

        # Use profit factor as the main score
        # Could also use: win_rate, total_profit, or a combination
        score = results["profit_factor"]

        return score

    return objective


# ============================================================
# STEP 5: Main Function
# ============================================================

def main():
    print("=" * 60)
    print("   PROOF OF CONCEPT: Self-Improving Parameter Optimizer")
    print("=" * 60)
    print()

    # Find data file
    data_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(data_dir, "data", "XAUJPY_H1_extended.csv")

    if not os.path.exists(csv_path):
        # Try alternative paths
        alternatives = [
            os.path.join(data_dir, "data", "XAUJPYH1.csv"),
            os.path.join(data_dir, "data", "USDJPYH1.csv"),
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                csv_path = alt
                break

    print(f"Loading data from: {csv_path}")
    prices = load_price_data(csv_path)

    if len(prices) < 200:
        print("Not enough data for backtesting. Need at least 200 candles.")
        return

    # Show current parameters (from your system)
    print("\n" + "=" * 60)
    print("CURRENT PARAMETERS (hardcoded in your system):")
    print("=" * 60)
    current_params = {
        "adx_threshold": 5,
        "slope_threshold": 0.00001,
        "tp_mult": 2.0,
        "sl_mult": 1.5,
        "rsi_buy_max": 75,
        "rsi_sell_min": 25,
    }
    for k, v in current_params.items():
        print(f"  {k}: {v}")

    current_results = run_backtest(prices, current_params)
    print(f"\nCurrent Performance:")
    print(f"  Trades: {current_results['num_trades']}")
    print(f"  Win Rate: {current_results['win_rate']:.1f}%")
    print(f"  Profit Factor: {current_results['profit_factor']:.2f}")
    print(f"  Total Profit: {current_results['total_profit']:.0f}")

    # Run Optuna optimization
    print("\n" + "=" * 60)
    print("RUNNING OPTUNA OPTIMIZATION...")
    print("(Trying 50 different parameter combinations)")
    print("=" * 60)

    study = optuna.create_study(direction="maximize")
    objective = create_objective(prices)

    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # Show results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 60)

    best_params = study.best_params
    print("\nBEST PARAMETERS FOUND:")
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    # Run backtest with best params
    best_results = run_backtest(prices, best_params)
    print(f"\nOptimized Performance:")
    print(f"  Trades: {best_results['num_trades']}")
    print(f"  Win Rate: {best_results['win_rate']:.1f}%")
    print(f"  Profit Factor: {best_results['profit_factor']:.2f}")
    print(f"  Total Profit: {best_results['total_profit']:.0f}")

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON: Current vs Optimized")
    print("=" * 60)
    print(f"  Profit Factor: {current_results['profit_factor']:.2f} -> {best_results['profit_factor']:.2f}")
    print(f"  Win Rate: {current_results['win_rate']:.1f}% -> {best_results['win_rate']:.1f}%")
    print(f"  Trades: {current_results['num_trades']} -> {best_results['num_trades']}")

    improvement = ((best_results['profit_factor'] - current_results['profit_factor'])
                   / current_results['profit_factor'] * 100) if current_results['profit_factor'] > 0 else 0
    print(f"\n  Improvement: {improvement:+.1f}%")

    print("\n" + "=" * 60)
    print("This is just a DEMO. The full system will:")
    print("  - Use AI to analyze WHY these parameters work better")
    print("  - Apply safety limits to prevent wild changes")
    print("  - Run automatically every day")
    print("  - Update your live trading config")
    print("=" * 60)


if __name__ == "__main__":
    main()
