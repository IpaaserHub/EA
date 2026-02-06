"""
Optimizer Scheduler
===================
Schedules automatic parameter optimization runs using APScheduler.

This module:
1. Schedules daily/weekly optimization runs
2. Logs results to files
3. Sends notifications on improvements
4. Provides CLI for managing scheduler
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)


class OptimizerScheduler:
    """
    Schedules automatic optimization runs.

    Usage:
        scheduler = OptimizerScheduler(
            optimization_loop,
            symbols=["XAUJPY", "BTCJPY"]
        )
        scheduler.start()  # Runs in background
        # ... do other work ...
        scheduler.stop()
    """

    def __init__(
        self,
        optimization_loop,
        symbols: Optional[List[str]] = None,
        log_dir: str = "logs/optimizer",
        schedule_hour: int = 2,  # Run at 2 AM by default
        schedule_minute: int = 0,
        n_trials: int = 50,
        auto_apply: bool = False,
    ):
        """
        Initialize scheduler.

        Args:
            optimization_loop: OptimizationLoop instance
            symbols: Symbols to optimize (None = auto-detect)
            log_dir: Directory for optimization logs
            schedule_hour: Hour to run (0-23)
            schedule_minute: Minute to run (0-59)
            n_trials: Number of optimization trials
            auto_apply: Whether to auto-apply improvements
        """
        self.optimization_loop = optimization_loop
        self.symbols = symbols
        self.log_dir = log_dir
        self.schedule_hour = schedule_hour
        self.schedule_minute = schedule_minute
        self.n_trials = n_trials
        self.auto_apply = auto_apply
        self._scheduler = None
        self._running = False

        os.makedirs(log_dir, exist_ok=True)

    def start(self) -> bool:
        """
        Start the scheduler.

        Returns:
            True if started successfully
        """
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            logger.error("APScheduler not installed. Install with: pip install apscheduler")
            return False

        if self._running:
            logger.warning("Scheduler already running")
            return False

        self._scheduler = BackgroundScheduler()

        # Schedule daily optimization
        trigger = CronTrigger(
            hour=self.schedule_hour,
            minute=self.schedule_minute,
        )

        self._scheduler.add_job(
            self._run_optimization,
            trigger,
            id="daily_optimization",
            name="Daily Parameter Optimization",
            replace_existing=True,
        )

        self._scheduler.start()
        self._running = True

        logger.info(
            f"Scheduler started. "
            f"Optimization runs at {self.schedule_hour:02d}:{self.schedule_minute:02d} daily"
        )
        return True

    def stop(self) -> bool:
        """
        Stop the scheduler.

        Returns:
            True if stopped successfully
        """
        if not self._running or not self._scheduler:
            logger.warning("Scheduler not running")
            return False

        self._scheduler.shutdown(wait=True)
        self._running = False
        logger.info("Scheduler stopped")
        return True

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    def run_now(self) -> List[Dict[str, Any]]:
        """
        Run optimization immediately (manual trigger).

        Returns:
            List of optimization results
        """
        return self._run_optimization()

    def _run_optimization(self) -> List[Dict[str, Any]]:
        """Run optimization for all configured symbols."""
        start_time = datetime.now()
        logger.info(f"Starting scheduled optimization at {start_time}")

        results = []

        symbols = self.symbols
        if not symbols:
            # Auto-detect from optimization loop
            config_dir = self.optimization_loop.config_dir
            if os.path.exists(config_dir):
                symbols = [
                    f.replace('.json', '')
                    for f in os.listdir(config_dir)
                    if f.endswith('.json') and not f.startswith('optimization')
                ]

        for symbol in symbols:
            try:
                run = self.optimization_loop.optimize_symbol(
                    symbol,
                    n_trials=self.n_trials,
                    auto_apply=self.auto_apply,
                )

                result = {
                    "symbol": symbol,
                    "status": "success",
                    "improvement_pct": run.improvement_pct,
                    "applied": run.applied,
                    "reason": run.reason,
                    "old_profit_factor": run.old_result.profit_factor,
                    "new_profit_factor": run.new_result.profit_factor,
                }
                results.append(result)

                logger.info(
                    f"Optimized {symbol}: "
                    f"PF {run.old_result.profit_factor:.2f} -> {run.new_result.profit_factor:.2f} "
                    f"({run.improvement_pct:+.1f}%) "
                    f"{'[APPLIED]' if run.applied else '[NOT APPLIED]'}"
                )

            except Exception as e:
                logger.error(f"Failed to optimize {symbol}: {e}")
                results.append({
                    "symbol": symbol,
                    "status": "error",
                    "error": str(e),
                })

        # Save log
        self._save_log(results, start_time)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Optimization completed in {duration:.1f}s")

        return results

    def _save_log(self, results: List[Dict[str, Any]], timestamp: datetime):
        """Save optimization results to log file."""
        log_file = os.path.join(
            self.log_dir,
            f"optimization_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        )

        log_data = {
            "timestamp": timestamp.isoformat(),
            "n_trials": self.n_trials,
            "auto_apply": self.auto_apply,
            "results": results,
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Saved log to {log_file}")

    def get_next_run_time(self) -> Optional[datetime]:
        """Get the next scheduled run time."""
        if not self._scheduler or not self._running:
            return None

        job = self._scheduler.get_job("daily_optimization")
        if job:
            return job.next_run_time
        return None

    def get_recent_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent optimization logs.

        Args:
            limit: Maximum number of logs to return

        Returns:
            List of log entries (newest first)
        """
        if not os.path.exists(self.log_dir):
            return []

        log_files = sorted(
            [f for f in os.listdir(self.log_dir) if f.endswith('.json')],
            reverse=True
        )[:limit]

        logs = []
        for filename in log_files:
            filepath = os.path.join(self.log_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    logs.append(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to read log {filename}: {e}")

        return logs


def create_scheduler(
    data_dir: str = "data",
    config_dir: str = "config/params",
    log_dir: str = "logs/optimizer",
    symbols: Optional[List[str]] = None,
    schedule_hour: int = 2,
    schedule_minute: int = 0,
    n_trials: int = 50,
    auto_apply: bool = False,
    use_ai: bool = False,
) -> OptimizerScheduler:
    """
    Convenience function to create a scheduler.

    Args:
        data_dir: Directory with price data
        config_dir: Directory with parameter configs
        log_dir: Directory for logs
        symbols: Symbols to optimize
        schedule_hour: Hour to run daily
        schedule_minute: Minute to run daily
        n_trials: Number of optimization trials
        auto_apply: Whether to auto-apply improvements
        use_ai: Whether to use AI analysis

    Returns:
        Configured OptimizerScheduler
    """
    from optimizer.optimization_loop import OptimizationLoop

    loop = OptimizationLoop(
        data_dir=data_dir,
        config_dir=config_dir,
        use_ai=use_ai,
    )

    return OptimizerScheduler(
        optimization_loop=loop,
        symbols=symbols,
        log_dir=log_dir,
        schedule_hour=schedule_hour,
        schedule_minute=schedule_minute,
        n_trials=n_trials,
        auto_apply=auto_apply,
    )


# ==================== CLI Interface ====================

def run_cli():
    """Command-line interface for the scheduler."""
    import argparse

    parser = argparse.ArgumentParser(
        description="EA Parameter Optimizer Scheduler"
    )
    parser.add_argument(
        "command",
        choices=["run", "schedule", "logs"],
        help="Command to execute"
    )
    parser.add_argument(
        "--symbols", "-s",
        nargs="+",
        help="Symbols to optimize"
    )
    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=50,
        help="Number of optimization trials"
    )
    parser.add_argument(
        "--auto-apply", "-a",
        action="store_true",
        help="Automatically apply improvements"
    )
    parser.add_argument(
        "--hour",
        type=int,
        default=2,
        help="Hour to run scheduled optimization (0-23)"
    )
    parser.add_argument(
        "--use-ai",
        action="store_true",
        help="Use AI analysis"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing price data"
    )
    parser.add_argument(
        "--config-dir",
        default="config/params",
        help="Directory containing parameter configs"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    scheduler = create_scheduler(
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        symbols=args.symbols,
        schedule_hour=args.hour,
        n_trials=args.trials,
        auto_apply=args.auto_apply,
        use_ai=args.use_ai,
    )

    if args.command == "run":
        # Run optimization immediately
        print(f"Running optimization for {args.symbols or 'all symbols'}...")
        results = scheduler.run_now()

        print("\n" + "=" * 60)
        print("OPTIMIZATION RESULTS")
        print("=" * 60)

        for r in results:
            if r["status"] == "success":
                status_str = "[APPLIED]" if r["applied"] else f"[NOT APPLIED: {r['reason']}]"
                print(
                    f"{r['symbol']}: "
                    f"PF {r['old_profit_factor']:.2f} -> {r['new_profit_factor']:.2f} "
                    f"({r['improvement_pct']:+.1f}%) "
                    f"{status_str}"
                )
            else:
                print(f"{r['symbol']}: ERROR - {r.get('error', 'Unknown')}")

    elif args.command == "schedule":
        # Start scheduler (blocking)
        print(f"Starting scheduler. Will run at {args.hour:02d}:00 daily.")
        print("Press Ctrl+C to stop.")

        scheduler.start()

        try:
            import time
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nStopping scheduler...")
            scheduler.stop()
            print("Scheduler stopped.")

    elif args.command == "logs":
        # Show recent logs
        logs = scheduler.get_recent_logs(limit=5)

        if not logs:
            print("No optimization logs found.")
            return

        print("\n" + "=" * 60)
        print("RECENT OPTIMIZATION LOGS")
        print("=" * 60)

        for log in logs:
            print(f"\n{log['timestamp']}")
            print(f"Trials: {log['n_trials']}, Auto-apply: {log['auto_apply']}")
            for r in log['results']:
                if r['status'] == 'success':
                    print(f"  {r['symbol']}: {r['improvement_pct']:+.1f}% {'[APPLIED]' if r['applied'] else ''}")
                else:
                    print(f"  {r['symbol']}: ERROR")


if __name__ == "__main__":
    run_cli()
