"""
AI EA Backtest Script
Test AI exit decisions with simulated price data
Supports multiple currency pairs with parallel testing
"""
import requests
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

SERVER_URL = "http://127.0.0.1:8000"
ACCOUNT_ID = 75449373

# Investment assumptions
INVESTMENT_CONFIG = {
    "initial_balance": 1000000,  # 100万円
    "lot_size": 0.01,            # 0.01ロット
    "trades_per_day": 5,         # 1日あたりのトレード数想定
}

# Currency pair configurations
SYMBOL_CONFIG = {
    "BTCJPY": {
        "base_price": 14000000,
        "tick_size": 5000,
        "noise_range": 2000,
        "range_noise": 50000,
        "profit_multiplier": 10,  # 1 tick = 10 JPY profit
        "avg_profit_per_win": 3000,   # 勝ちトレード平均利益
        "avg_loss_per_loss": -2000,   # 負けトレード平均損失
    },
    "USDJPY": {
        "base_price": 157.50,
        "tick_size": 0.05,
        "noise_range": 0.02,
        "range_noise": 0.50,
        "profit_multiplier": 100,
        "avg_profit_per_win": 2000,
        "avg_loss_per_loss": -1500,
    },
    "XAUUSD": {
        "base_price": 2650.00,
        "tick_size": 1.0,
        "noise_range": 0.5,
        "range_noise": 10.0,
        "profit_multiplier": 100,
        "avg_profit_per_win": 2500,
        "avg_loss_per_loss": -1800,
    }
}

def generate_prices(symbol, trend="up", bars=100):
    """Generate price data for any symbol"""
    config = SYMBOL_CONFIG.get(symbol, SYMBOL_CONFIG["BTCJPY"])
    base = config["base_price"]
    tick = config["tick_size"]
    noise = config["noise_range"]
    range_noise = config["range_noise"]

    prices = []
    for i in range(bars):
        if trend == "up":
            price = base + (i * tick) + random.uniform(-noise, noise)
        elif trend == "down":
            price = base - (i * tick) + random.uniform(-noise, noise)
        else:  # range
            price = base + random.uniform(-range_noise, range_noise)
        prices.append(round(price, 5))
    return prices

def send_history(symbol, prices):
    """Send price history to server"""
    resp = requests.post(f"{SERVER_URL}/history", json={
        "account_id": ACCOUNT_ID,
        "symbol": symbol,
        "prices": prices
    })
    return resp.json()

def test_exit_decision(symbol, position_type, open_price, current_price, profit, holding_minutes, sl, tp):
    """Test AI exit decision"""
    resp = requests.post(f"{SERVER_URL}/check_exit", json={
        "account_id": ACCOUNT_ID,
        "ticket": random.randint(1, 99999),
        "symbol": symbol,
        "position_type": position_type,
        "open_price": open_price,
        "current_price": current_price,
        "profit": profit,
        "volume": 0.01,
        "open_time": int(time.time()) - (holding_minutes * 60),
        "sl": sl,
        "tp": tp
    })
    return resp.json()

def get_test_cases(symbol, count=100):
    """Generate random test cases for a specific symbol"""
    config = SYMBOL_CONFIG.get(symbol, SYMBOL_CONFIG["BTCJPY"])
    base = config["base_price"]
    tick = config["tick_size"]

    test_cases = []

    for i in range(count):
        # Randomize conditions
        trend = random.choice(["up", "down", "range"])
        position_type = random.choice(["BUY", "SELL"])

        # Price movement (positive = price went up from open)
        price_move = random.randint(-80, 80) * tick
        current_price = base + price_move

        # Calculate profit based on position type and price movement
        if position_type == "BUY":
            profit = int(price_move * config["profit_multiplier"] / tick)
        else:
            profit = int(-price_move * config["profit_multiplier"] / tick)

        holding_minutes = random.randint(5, 180)

        # SL/TP settings (改善: SLは狭く、TPは広く = 損小利大)
        sl_distance = random.randint(10, 30) * tick   # SL狭く: 20-50 → 10-30
        tp_distance = random.randint(50, 120) * tick  # TP広く: 30-80 → 50-120

        if position_type == "BUY":
            sl = base - sl_distance
            tp = base + tp_distance
        else:
            sl = base + sl_distance
            tp = base - tp_distance

        # Determine expected outcome based on logic
        # This is what a rational trader would do
        expected = determine_expected_action(trend, position_type, profit, config["profit_multiplier"])

        test_cases.append({
            "name": f"[{symbol}] #{i+1}",
            "symbol": symbol,
            "trend": trend,
            "position_type": position_type,
            "open_price": base,
            "current_price": current_price,
            "profit": profit,
            "holding_minutes": holding_minutes,
            "sl": sl,
            "tp": tp,
            "expected": expected
        })

    return test_cases

def determine_expected_action(trend, position_type, profit, multiplier):
    """Determine the expected action based on trading logic

    Args:
        trend: "up", "down", or "range"
        position_type: "BUY" or "SELL"
        profit: Current profit in JPY
        multiplier: profit_multiplier for the symbol (to normalize thresholds)
    """
    # Trend alignment check
    trend_aligned = (
        (trend == "up" and position_type == "BUY") or
        (trend == "down" and position_type == "SELL")
    )
    trend_against = (
        (trend == "up" and position_type == "SELL") or
        (trend == "down" and position_type == "BUY")
    )

    # Relative thresholds based on multiplier (normalized to tick units)
    # 50 ticks = large profit, 30 ticks = large loss, 20 ticks = small profit
    large_profit = profit > 50 * multiplier   # 50+ ticks
    large_loss = profit < -30 * multiplier    # -30+ ticks
    small_profit = 0 < profit < 20 * multiplier  # 0-20 ticks
    medium_loss = -20 * multiplier < profit < 0  # -20 to 0 ticks

    # Decision logic (matching AI prompt priority order)
    # Priority 1: Loss + Against trend = CLOSE
    if trend_against and profit < 0:
        return "CLOSE"

    # Priority 2: Profit + With trend = HOLD (let it run)
    elif trend_aligned and profit > 0:
        if large_profit:
            return "EITHER"  # Large profit with trend - could take profit or hold
        else:
            return "HOLD"  # Small/medium profit with trend - hold for more

    # Priority 3: Large loss (regardless of trend) = CLOSE
    elif large_loss:
        return "CLOSE"

    # Priority 4: Range market = EITHER (no clear direction)
    elif trend == "range":
        return "EITHER"

    # Priority 5: Against trend but not losing much = CLOSE (risky position)
    elif trend_against:
        return "CLOSE"

    # Default: EITHER (unclear situation)
    else:
        return "EITHER"

def run_single_test(tc):
    """Run a single test case (for parallel execution)"""
    symbol = tc["symbol"]

    # Send price history
    prices = generate_prices(symbol, trend=tc["trend"], bars=100)
    send_history(symbol, prices)
    time.sleep(0.3)

    # Get AI decision
    result = test_exit_decision(
        symbol=symbol,
        position_type=tc["position_type"],
        open_price=tc["open_price"],
        current_price=tc["current_price"],
        profit=tc["profit"],
        holding_minutes=tc["holding_minutes"],
        sl=tc["sl"],
        tp=tc["tp"]
    )

    return {
        "case": tc["name"],
        "symbol": symbol,
        "trend": tc["trend"],
        "position": tc["position_type"],
        "profit": tc["profit"],
        "expected": tc["expected"],
        "action": result.get("action", "ERROR"),
        "reason": result.get("reason", "N/A")
    }

def print_preconditions(symbols, test_cases_count):
    """Print test preconditions"""
    print("=" * 70)
    print("[TEST PRECONDITIONS]")
    print("=" * 70)
    print(f"  Date/Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Server       : {SERVER_URL}")
    print(f"  Account ID   : {ACCOUNT_ID}")
    print(f"  Symbols      : {', '.join(symbols)}")
    print(f"  Test Cases   : {test_cases_count} total ({test_cases_count // len(symbols)} per symbol)")
    print(f"  Price Bars   : 100 bars per test")
    print(f"  Parallel     : {'Yes' if len(symbols) > 1 else 'No'}")
    print("")
    print("[INVESTMENT ASSUMPTIONS]")
    print(f"  Initial Balance : {INVESTMENT_CONFIG['initial_balance']:,} JPY")
    print(f"  Lot Size        : {INVESTMENT_CONFIG['lot_size']} lot")
    print(f"  Trades/Day      : {INVESTMENT_CONFIG['trades_per_day']} trades")
    print("")
    print("[SYMBOL CONFIGS]")
    for sym in symbols:
        cfg = SYMBOL_CONFIG[sym]
        print(f"  {sym:10} : Base={cfg['base_price']:>12,.2f}, Win={cfg['avg_profit_per_win']:+,}, Loss={cfg['avg_loss_per_loss']:,}")
    print("=" * 70)

def run_backtest(symbols=None, parallel=True):
    """Run backtest for specified symbols"""
    if symbols is None:
        symbols = ["BTCJPY"]

    # Collect all test cases
    all_test_cases = []
    for symbol in symbols:
        all_test_cases.extend(get_test_cases(symbol))

    # Print preconditions
    print_preconditions(symbols, len(all_test_cases))

    results = []

    if parallel and len(symbols) > 1:
        # Parallel execution
        print(f"\n[RUNNING {len(all_test_cases)} TESTS IN PARALLEL...]")
        completed = 0
        with ThreadPoolExecutor(max_workers=len(symbols)) as executor:
            futures = {executor.submit(run_single_test, tc): tc for tc in all_test_cases}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                # Show progress every 10 tests
                if completed % 10 == 0 or completed == len(all_test_cases):
                    print(f"  Progress: {completed}/{len(all_test_cases)} ({completed/len(all_test_cases)*100:.0f}%)")
                time.sleep(0.3)  # API rate limit
    else:
        # Sequential execution
        for i, tc in enumerate(all_test_cases, 1):
            print(f"\n{'-' * 70}")
            print(f"[TEST {i}/{len(all_test_cases)}] {tc['name']}")
            print(f"{'-' * 70}")

            result = run_single_test(tc)
            results.append(result)

            print(f"  Trend: {tc['trend']:6} | Position: {tc['position_type']} | P/L: {tc['profit']:+,}")
            print(f"  AI Decision : {result['action']} - {result['reason']}")
            print(f"  Expected    : {result['expected']}")

            time.sleep(1)  # API rate limit

    # Results summary
    print(f"\n{'=' * 70}")
    print("[RESULTS SUMMARY]")
    print("=" * 70)

    # Group by symbol
    for symbol in symbols:
        symbol_results = [r for r in results if r["symbol"] == symbol]
        cfg = SYMBOL_CONFIG[symbol]

        # Count statistics
        ok_count = sum(1 for r in symbol_results if r["expected"] == r["action"] or r["expected"] == "EITHER")
        hold_count = sum(1 for r in symbol_results if r["action"] == "HOLD")
        close_count = sum(1 for r in symbol_results if r["action"] == "CLOSE")

        # Count by expected
        expected_hold = sum(1 for r in symbol_results if r["expected"] == "HOLD")
        expected_close = sum(1 for r in symbol_results if r["expected"] == "CLOSE")
        expected_either = sum(1 for r in symbol_results if r["expected"] == "EITHER")

        # Mismatches
        false_close = sum(1 for r in symbol_results if r["expected"] == "HOLD" and r["action"] == "CLOSE")
        false_hold = sum(1 for r in symbol_results if r["expected"] == "CLOSE" and r["action"] == "HOLD")

        # Per-symbol P/L statistics
        sym_total_profit = sum(r["profit"] for r in symbol_results)
        sym_winning = sum(1 for r in symbol_results if r["profit"] > 0)
        sym_losing = sum(1 for r in symbol_results if r["profit"] < 0)
        sym_win_rate = sym_winning / len(symbol_results) * 100 if symbol_results else 0
        sym_avg_win = sum(r["profit"] for r in symbol_results if r["profit"] > 0) / sym_winning if sym_winning > 0 else 0
        sym_avg_loss = sum(r["profit"] for r in symbol_results if r["profit"] < 0) / sym_losing if sym_losing > 0 else 0
        sym_total_wins = sum(r["profit"] for r in symbol_results if r["profit"] > 0)
        sym_total_losses = abs(sum(r["profit"] for r in symbol_results if r["profit"] < 0))
        sym_pf = sym_total_wins / sym_total_losses if sym_total_losses > 0 else float('inf')

        print(f"\n  [{symbol}] Accuracy: {ok_count}/{len(symbol_results)} ({ok_count/len(symbol_results)*100:.1f}%)")
        print(f"    AI Decisions  : HOLD={hold_count}, CLOSE={close_count}")
        print(f"    Expected      : HOLD={expected_hold}, CLOSE={expected_close}, EITHER={expected_either}")
        print(f"    Errors        : FalseClose={false_close}, FalseHold={false_hold}")
        print(f"    --- P/L Stats ---")
        print(f"    Win Rate      : {sym_win_rate:.1f}% ({sym_winning}W / {sym_losing}L)")
        print(f"    Avg Win/Loss  : {sym_avg_win:+,.0f} / {sym_avg_loss:+,.0f} JPY")
        print(f"    Profit Factor : {sym_pf:.2f}")
        print(f"    Total P/L     : {sym_total_profit:+,.0f} JPY")

    # Overall stats
    total_ok = sum(1 for r in results if r["expected"] == r["action"] or r["expected"] == "EITHER")
    total_hold = sum(1 for r in results if r["action"] == "HOLD")
    total_close = sum(1 for r in results if r["action"] == "CLOSE")
    total_false_close = sum(1 for r in results if r["expected"] == "HOLD" and r["action"] == "CLOSE")
    total_false_hold = sum(1 for r in results if r["expected"] == "CLOSE" and r["action"] == "HOLD")

    # Calculate profit/loss simulation
    total_profit = sum(r["profit"] for r in results)
    winning_trades = sum(1 for r in results if r["profit"] > 0)
    losing_trades = sum(1 for r in results if r["profit"] < 0)
    win_rate = winning_trades / len(results) * 100 if results else 0

    # Average profit/loss
    avg_win = sum(r["profit"] for r in results if r["profit"] > 0) / winning_trades if winning_trades > 0 else 0
    avg_loss = sum(r["profit"] for r in results if r["profit"] < 0) / losing_trades if losing_trades > 0 else 0

    # Profit factor
    total_wins = sum(r["profit"] for r in results if r["profit"] > 0)
    total_losses = abs(sum(r["profit"] for r in results if r["profit"] < 0))
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

    # Projected returns
    trades_per_day = INVESTMENT_CONFIG["trades_per_day"]
    avg_profit_per_trade = total_profit / len(results) if results else 0
    daily_profit = avg_profit_per_trade * trades_per_day
    monthly_profit = daily_profit * 20  # 20 trading days
    initial_balance = INVESTMENT_CONFIG["initial_balance"]
    monthly_return_pct = (monthly_profit / initial_balance) * 100

    print(f"\n{'=' * 70}")
    print(f"[TOTAL RESULTS]")
    print(f"  Tests        : {len(results)}")
    print(f"  AI Accuracy  : {total_ok}/{len(results)} ({total_ok/len(results)*100:.1f}%)")
    print(f"  AI Output    : HOLD={total_hold} ({total_hold/len(results)*100:.1f}%), CLOSE={total_close} ({total_close/len(results)*100:.1f}%)")
    print(f"  Errors       : FalseClose={total_false_close}, FalseHold={total_false_hold}")
    print("")
    print(f"[PROFIT/LOSS SIMULATION]")
    print(f"  Win Rate     : {win_rate:.1f}% ({winning_trades}W / {losing_trades}L)")
    print(f"  Avg Win      : {avg_win:+,.0f} JPY")
    print(f"  Avg Loss     : {avg_loss:+,.0f} JPY")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Total P/L    : {total_profit:+,.0f} JPY (from {len(results)} trades)")
    print("")
    print(f"[PROJECTED RETURNS] (Based on {trades_per_day} trades/day)")
    print(f"  Initial      : {initial_balance:,} JPY")
    print(f"  Daily P/L    : {daily_profit:+,.0f} JPY")
    print(f"  Monthly P/L  : {monthly_profit:+,.0f} JPY")
    print(f"  Monthly ROI  : {monthly_return_pct:+.1f}%")
    print("=" * 70)

    # Improvement suggestions
    print_improvement_suggestions(results, symbols, total_false_close, total_false_hold, profit_factor)

def print_improvement_suggestions(results, symbols, total_false_close, total_false_hold, profit_factor):
    """Analyze results and print improvement suggestions"""
    print(f"\n{'=' * 70}")
    print("[IMPROVEMENT SUGGESTIONS]")
    print("=" * 70)

    suggestions = []

    # 1. Analyze FalseClose (closing when should hold)
    if total_false_close > 30:
        # Find which conditions cause FalseClose
        false_close_cases = [r for r in results if r["expected"] == "HOLD" and r["action"] == "CLOSE"]
        fc_with_profit = sum(1 for r in false_close_cases if r["profit"] > 0)
        fc_trend_aligned = sum(1 for r in false_close_cases if
            (r["trend"] == "up" and r["position"] == "BUY") or
            (r["trend"] == "down" and r["position"] == "SELL"))

        if fc_with_profit > len(false_close_cases) * 0.5:
            suggestions.append({
                "issue": "FalseClose (Early Profit Taking)",
                "count": fc_with_profit,
                "suggestion": "AI is closing profitable positions too early. Consider:\n"
                             "    - Increase HOLD threshold for trend-aligned profits\n"
                             "    - Add rule: 'Profit + Trend aligned -> HOLD unless RSI extreme'"
            })
        if fc_trend_aligned > len(false_close_cases) * 0.5:
            suggestions.append({
                "issue": "FalseClose (Trend-Aligned Positions)",
                "count": fc_trend_aligned,
                "suggestion": "AI closes trend-aligned positions prematurely. Consider:\n"
                             "    - Strengthen Priority 2 rule in prompt\n"
                             "    - Add minimum profit target before considering CLOSE"
            })

    # 2. Analyze FalseHold (holding when should close)
    if total_false_hold > 30:
        false_hold_cases = [r for r in results if r["expected"] == "CLOSE" and r["action"] == "HOLD"]
        fh_with_loss = sum(1 for r in false_hold_cases if r["profit"] < 0)
        fh_trend_against = sum(1 for r in false_hold_cases if
            (r["trend"] == "up" and r["position"] == "SELL") or
            (r["trend"] == "down" and r["position"] == "BUY"))

        if fh_with_loss > len(false_hold_cases) * 0.5:
            suggestions.append({
                "issue": "FalseHold (Late Loss Cutting)",
                "count": fh_with_loss,
                "suggestion": "AI holds losing positions too long. Consider:\n"
                             "    - Make Priority 1 (loss + against trend) stricter\n"
                             "    - Add rule: 'Loss > X yen -> CLOSE regardless of trend'"
            })
        if fh_trend_against > len(false_hold_cases) * 0.5:
            suggestions.append({
                "issue": "FalseHold (Against-Trend Positions)",
                "count": fh_trend_against,
                "suggestion": "AI holds against-trend positions. Consider:\n"
                             "    - Add explicit rule for against-trend with small profit -> CLOSE\n"
                             "    - Reduce HOLD tendency in Priority 4"
            })

    # 3. Per-symbol analysis
    for symbol in symbols:
        sym_results = [r for r in results if r["symbol"] == symbol]
        sym_wins = sum(r["profit"] for r in sym_results if r["profit"] > 0)
        sym_losses = abs(sum(r["profit"] for r in sym_results if r["profit"] < 0))
        sym_pf = sym_wins / sym_losses if sym_losses > 0 else float('inf')

        if sym_pf < 1.0:
            sym_false_hold = sum(1 for r in sym_results if r["expected"] == "CLOSE" and r["action"] == "HOLD")
            sym_false_close = sum(1 for r in sym_results if r["expected"] == "HOLD" and r["action"] == "CLOSE")

            if sym_false_hold > sym_false_close:
                suggestions.append({
                    "issue": f"{symbol}: Low PF ({sym_pf:.2f})",
                    "count": sym_false_hold,
                    "suggestion": f"{symbol} has high FalseHold ({sym_false_hold}). Consider:\n"
                                 f"    - Symbol-specific stricter loss-cutting rules\n"
                                 f"    - Lower threshold for CLOSE decision on {symbol}"
                })
            else:
                suggestions.append({
                    "issue": f"{symbol}: Low PF ({sym_pf:.2f})",
                    "count": sym_false_close,
                    "suggestion": f"{symbol} has high FalseClose ({sym_false_close}). Consider:\n"
                                 f"    - Symbol-specific HOLD rules\n"
                                 f"    - Higher profit threshold before CLOSE on {symbol}"
                })

    # 4. Overall profit factor analysis
    if profit_factor < 1.0:
        suggestions.append({
            "issue": "Overall PF < 1.0 (Losing System)",
            "count": 0,
            "suggestion": "System is not profitable. Priority fixes:\n"
                         "    1. Reduce FalseHold - cut losses faster\n"
                         "    2. Reduce FalseClose - let winners run\n"
                         "    3. Consider removing worst-performing symbol"
        })
    elif profit_factor < 1.2:
        suggestions.append({
            "issue": "Marginal PF (1.0-1.2)",
            "count": 0,
            "suggestion": "System is marginally profitable. To improve:\n"
                         "    1. Fine-tune RSI thresholds (current 75/25)\n"
                         "    2. Add holding time factor to decisions\n"
                         "    3. Consider ATR-based profit targets"
        })

    # Print suggestions
    if suggestions:
        for i, s in enumerate(suggestions, 1):
            print(f"\n  [{i}] {s['issue']}" + (f" ({s['count']} cases)" if s['count'] > 0 else ""))
            print(f"      {s['suggestion']}")
    else:
        print("\n  No critical issues found. System performing well.")

    # Summary action items
    print(f"\n{'=' * 70}")
    print("[PRIORITY ACTIONS]")
    print("=" * 70)

    if total_false_hold > total_false_close:
        print("  1. PRIORITY: Reduce FalseHold (loss-cutting too slow)")
        print("     -> Strengthen 'Loss + Against trend = CLOSE' rule")
    else:
        print("  1. PRIORITY: Reduce FalseClose (taking profits too early)")
        print("     -> Strengthen 'Profit + With trend = HOLD' rule")

    worst_symbol = min(symbols, key=lambda s:
        sum(r["profit"] for r in results if r["symbol"] == s))
    worst_pnl = sum(r["profit"] for r in results if r["symbol"] == worst_symbol)
    print(f"  2. Focus on: {worst_symbol} (Total P/L: {worst_pnl:+,.0f} JPY)")

    print("=" * 70)

if __name__ == "__main__":
    # Single symbol test
    # run_backtest(["BTCJPY"])

    # Multiple symbols in parallel
    run_backtest(["BTCJPY", "USDJPY", "XAUUSD"], parallel=True)
