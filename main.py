#!/usr/bin/env python3
"""
EA Parameter Optimizer - Main Entry Point
==========================================
Self-improving parameter optimization system for MetaTrader 5 Expert Advisors.

Usage:
    python main.py optimize SYMBOL [--trials N] [--auto-apply]
    python main.py optimize-all [--trials N] [--auto-apply]
    python main.py schedule [--hour H]
    python main.py backtest SYMBOL [--params FILE]
    python main.py rollback SYMBOL [--steps N]
    python main.py status

Examples:
    python main.py optimize XAUJPY --trials 50
    python main.py optimize-all --auto-apply
    python main.py schedule --hour 2
    python main.py backtest XAUJPY
    python main.py rollback XAUJPY --steps 1
"""

import argparse
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.param_manager import ParamManager, PARAM_LIMITS
from optimizer.optimization_loop import (
    OptimizationLoop,
    print_run_summary,
)
from scheduler.optimizer_scheduler import create_scheduler


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_optimize(args):
    """Run optimization for a single symbol."""
    print(f"\n{'='*60}")
    print(f"Optimizing {args.symbol}")
    print(f"{'='*60}")
    print(f"Trials: {args.trials}")
    print(f"Auto-apply: {args.auto_apply}")
    print(f"Use AI: {args.use_ai}")
    print()

    loop = OptimizationLoop(
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        use_ai=args.use_ai,
    )

    try:
        run = loop.optimize_symbol(
            args.symbol,
            n_trials=args.trials,
            auto_apply=args.auto_apply,
        )
        print_run_summary(run)

        if not run.applied and run.reason == "Meets all criteria":
            print("\nTo apply these changes, run:")
            print(f"  python main.py optimize {args.symbol} --auto-apply")

        return 0

    except ValueError as e:
        print(f"Error: {e}")
        return 1


def cmd_optimize_all(args):
    """Run optimization for all symbols."""
    print(f"\n{'='*60}")
    print("Optimizing All Symbols")
    print(f"{'='*60}")

    loop = OptimizationLoop(
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        use_ai=args.use_ai,
    )

    runs = loop.optimize_all_symbols(
        n_trials=args.trials,
        auto_apply=args.auto_apply,
    )

    for run in runs:
        print_run_summary(run)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total symbols: {len(runs)}")
    applied = sum(1 for r in runs if r.applied)
    print(f"Applied: {applied}")
    print(f"Not applied: {len(runs) - applied}")

    return 0


def cmd_schedule(args):
    """Start the optimization scheduler."""
    print(f"\n{'='*60}")
    print("Starting Optimizer Scheduler")
    print(f"{'='*60}")
    print(f"Schedule: Daily at {args.hour:02d}:00")
    print(f"Trials per symbol: {args.trials}")
    print(f"Auto-apply: {args.auto_apply}")
    print(f"\nPress Ctrl+C to stop.\n")

    scheduler = create_scheduler(
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        schedule_hour=args.hour,
        n_trials=args.trials,
        auto_apply=args.auto_apply,
        use_ai=args.use_ai,
    )

    if not scheduler.start():
        print("Failed to start scheduler. Is APScheduler installed?")
        return 1

    try:
        import time
        while True:
            next_run = scheduler.get_next_run_time()
            if next_run:
                print(f"\rNext optimization: {next_run.strftime('%Y-%m-%d %H:%M:%S')}", end="")
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n\nStopping scheduler...")
        scheduler.stop()
        print("Scheduler stopped.")

    return 0


def cmd_backtest(args):
    """Run backtest with current parameters."""
    print(f"\n{'='*60}")
    print(f"Backtesting {args.symbol}")
    print(f"{'='*60}")

    from backtest.data_loader import load_prices, find_data_file
    from backtest.engine import BacktestEngine

    # Load parameters
    pm = ParamManager(args.config_dir)
    params = pm.load(args.symbol)

    print("\nParameters:")
    for key, value in sorted(params.items()):
        if key in PARAM_LIMITS:
            print(f"  {key}: {value}")

    # Load data
    data_file = find_data_file(args.symbol, args.data_dir)
    if not data_file:
        print(f"\nError: No data file found for {args.symbol}")
        return 1

    prices = load_prices(data_file)
    print(f"\nLoaded {len(prices)} candles from {os.path.basename(data_file)}")

    # Run backtest
    engine = BacktestEngine(prices)
    result = engine.run(params)

    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Wins: {result.wins}")
    print(f"Losses: {result.losses}")
    print(f"Win Rate: {result.win_rate:.1f}%")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Total Profit: {result.total_profit:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2f}")
    print(f"Avg Win: {result.avg_win:.2f}")
    print(f"Avg Loss: {result.avg_loss:.2f}")

    return 0


def cmd_rollback(args):
    """Rollback parameters to previous version."""
    print(f"\n{'='*60}")
    print(f"Rolling back {args.symbol}")
    print(f"{'='*60}")

    pm = ParamManager(args.config_dir)

    # Show history
    history = pm.get_history(args.symbol, limit=5)
    if not history:
        print("No history available for this symbol.")
        return 1

    print("\nRecent history:")
    for i, entry in enumerate(reversed(history)):
        print(f"  {i}: {entry['timestamp']} - {entry.get('reason', 'N/A')}")

    # Perform rollback
    result = pm.rollback(args.symbol, args.steps)
    if result:
        print(f"\nRolled back {args.steps} step(s).")
        print("Restored parameters:")
        for key, value in sorted(result.items()):
            if key in PARAM_LIMITS:
                print(f"  {key}: {value}")
        return 0
    else:
        print(f"\nCould not rollback {args.steps} step(s). Not enough history.")
        return 1


def cmd_status(args):
    """Show current status and configuration."""
    print(f"\n{'='*60}")
    print("EA Parameter Optimizer Status")
    print(f"{'='*60}")

    print(f"\nData directory: {args.data_dir}")
    print(f"Config directory: {args.config_dir}")

    # List symbols
    pm = ParamManager(args.config_dir)
    if os.path.exists(args.config_dir):
        symbols = [
            f.replace('.json', '')
            for f in os.listdir(args.config_dir)
            if f.endswith('.json') and not f.startswith('optimization')
        ]
        print(f"\nConfigured symbols: {len(symbols)}")
        for sym in sorted(symbols):
            params = pm.load(sym)
            print(f"  {sym}: ADX={params.get('adx_threshold')}, TP={params.get('tp_mult')}, SL={params.get('sl_mult')}")
    else:
        print("\nNo config directory found.")

    # Check data files
    from backtest.data_loader import find_data_file
    print(f"\nData files:")
    if os.path.exists(args.data_dir):
        for sym in sorted(symbols) if symbols else []:
            data_file = find_data_file(sym, args.data_dir)
            if data_file:
                print(f"  {sym}: {os.path.basename(data_file)}")
            else:
                print(f"  {sym}: NO DATA")
    else:
        print("  No data directory found.")

    # Recent optimization logs
    scheduler = create_scheduler(
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        use_ai=False,
    )
    logs = scheduler.get_recent_logs(limit=3)
    if logs:
        print(f"\nRecent optimization runs:")
        for log in logs:
            print(f"  {log['timestamp']}")
            for r in log['results']:
                if r['status'] == 'success':
                    print(f"    {r['symbol']}: {r['improvement_pct']:+.1f}% {'[APPLIED]' if r['applied'] else ''}")
                else:
                    print(f"    {r['symbol']}: ERROR")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EA Parameter Optimizer - Self-improving trading parameter optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing price data CSVs",
    )
    parser.add_argument(
        "--config-dir",
        default="config/params",
        help="Directory containing parameter JSON files",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # optimize command
    p_optimize = subparsers.add_parser("optimize", help="Optimize a single symbol")
    p_optimize.add_argument("symbol", help="Trading symbol (e.g., XAUJPY)")
    p_optimize.add_argument("-t", "--trials", type=int, default=50, help="Number of trials")
    p_optimize.add_argument("-a", "--auto-apply", action="store_true", help="Auto-apply improvements")
    p_optimize.add_argument("--use-ai", action="store_true", help="Use AI analysis")
    p_optimize.set_defaults(func=cmd_optimize)

    # optimize-all command
    p_optimize_all = subparsers.add_parser("optimize-all", help="Optimize all symbols")
    p_optimize_all.add_argument("-t", "--trials", type=int, default=50, help="Number of trials per symbol")
    p_optimize_all.add_argument("-a", "--auto-apply", action="store_true", help="Auto-apply improvements")
    p_optimize_all.add_argument("--use-ai", action="store_true", help="Use AI analysis")
    p_optimize_all.set_defaults(func=cmd_optimize_all)

    # schedule command
    p_schedule = subparsers.add_parser("schedule", help="Start optimization scheduler")
    p_schedule.add_argument("--hour", type=int, default=2, help="Hour to run (0-23)")
    p_schedule.add_argument("-t", "--trials", type=int, default=50, help="Number of trials per symbol")
    p_schedule.add_argument("-a", "--auto-apply", action="store_true", help="Auto-apply improvements")
    p_schedule.add_argument("--use-ai", action="store_true", help="Use AI analysis")
    p_schedule.set_defaults(func=cmd_schedule)

    # backtest command
    p_backtest = subparsers.add_parser("backtest", help="Run backtest with current params")
    p_backtest.add_argument("symbol", help="Trading symbol")
    p_backtest.set_defaults(func=cmd_backtest)

    # rollback command
    p_rollback = subparsers.add_parser("rollback", help="Rollback to previous parameters")
    p_rollback.add_argument("symbol", help="Trading symbol")
    p_rollback.add_argument("-s", "--steps", type=int, default=1, help="Steps to rollback")
    p_rollback.set_defaults(func=cmd_rollback)

    # status command
    p_status = subparsers.add_parser("status", help="Show current status")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    setup_logging(args.verbose)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
