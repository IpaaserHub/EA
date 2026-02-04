"""
Walk-Forward Optimization
=========================
Validates trading strategy robustness by testing on out-of-sample data.

This module:
1. Splits data into multiple time-based folds
2. Optimizes parameters on in-sample (training) data
3. Validates on out-of-sample (test) data
4. Calculates robustness metrics
"""

import logging
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Result from walk-forward analysis."""
    in_sample_results: List[Dict[str, Any]]   # Results from optimization periods
    out_sample_results: List[Dict[str, Any]]  # Results from validation periods
    avg_in_sample_pf: float                   # Average in-sample profit factor
    avg_out_sample_pf: float                  # Average out-of-sample profit factor
    robustness_ratio: float                   # out/in ratio (closer to 1.0 = better)
    is_robust: bool                           # True if strategy passes robustness test

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "in_sample_results": self.in_sample_results,
            "out_sample_results": self.out_sample_results,
            "avg_in_sample_pf": round(self.avg_in_sample_pf, 4),
            "avg_out_sample_pf": round(self.avg_out_sample_pf, 4),
            "robustness_ratio": round(self.robustness_ratio, 4),
            "is_robust": self.is_robust,
        }


class WalkForwardOptimizer:
    """
    Walk-forward optimization for trading strategy validation.

    Uses TimeSeriesSplit to divide data into folds, optimizes on
    in-sample data, and validates on out-of-sample data.

    Usage:
        wfo = WalkForwardOptimizer(n_splits=5)
        result = wfo.run(prices, backtest_fn, initial_params)
        print(f"Robustness: {result.robustness_ratio}")
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: int = 100,
        optuna_trials: int = 30,
        seed: int = None,
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            n_splits: Number of walk-forward folds
            min_train_size: Minimum candles for training period
            optuna_trials: Optuna trials per fold
            seed: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.optuna_trials = optuna_trials
        self.seed = seed

    def _split_data(self, prices: List[Dict]) -> List[tuple]:
        """
        Split price data into train/test folds using TimeSeriesSplit.

        Args:
            prices: List of OHLC dicts

        Returns:
            List of (train_data, test_data) tuples

        Raises:
            ValueError: If prices list is empty
        """
        if not prices:
            raise ValueError("prices list cannot be empty")

        n_samples = len(prices)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        folds = []
        for train_idx, test_idx in tscv.split(range(n_samples)):
            # Ensure minimum training size
            if len(train_idx) < self.min_train_size:
                continue

            train_data = [prices[i] for i in train_idx]
            test_data = [prices[i] for i in test_idx]
            folds.append((train_data, test_data))

        if not folds:
            logger.warning(f"No valid folds generated. Data has {n_samples} samples, min_train_size={self.min_train_size}")

        return folds

    def run(
        self,
        prices: List[Dict],
        backtest_fn: Callable,
        initial_params: Dict[str, Any],
        show_progress: bool = True,
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.

        Args:
            prices: List of OHLC price dicts
            backtest_fn: Function(prices, params) -> BacktestResult
            initial_params: Starting parameters
            show_progress: Print progress messages

        Returns:
            WalkForwardResult with robustness metrics
        """
        from .optuna_optimizer import OptunaOptimizer

        folds = self._split_data(prices)

        if not folds:
            logger.warning("Not enough data for walk-forward analysis")
            return WalkForwardResult(
                in_sample_results=[],
                out_sample_results=[],
                avg_in_sample_pf=0,
                avg_out_sample_pf=0,
                robustness_ratio=0,
                is_robust=False,
            )

        in_sample_results = []
        out_sample_results = []

        for fold_idx, (train_data, test_data) in enumerate(folds):
            if show_progress:
                logger.info(f"=== Fold {fold_idx + 1}/{len(folds)} ===")
                logger.info(f"  Train: {len(train_data)} candles, Test: {len(test_data)} candles")

            # Create optimizer for this fold
            optimizer = OptunaOptimizer(
                study_name=f"wfo_fold_{fold_idx}",
                seed=self.seed,
            )

            # Objective: maximize profit factor on training data
            def objective(params):
                result = backtest_fn(train_data, params)
                return result.profit_factor if result.profit_factor else 0.0

            # Optimize on in-sample data
            opt_result = optimizer.optimize(
                objective_fn=objective,
                n_trials=self.optuna_trials,
                initial_params=initial_params,
                show_progress=False,
            )

            best_params = opt_result.best_params

            # Backtest on in-sample with best params
            in_result = backtest_fn(train_data, best_params)
            in_sample_results.append({
                "fold": fold_idx,
                "profit_factor": in_result.profit_factor,
                "win_rate": in_result.win_rate,
                "total_trades": in_result.total_trades,
                "params": best_params,
            })

            # Validate on out-of-sample data
            out_result = backtest_fn(test_data, best_params)
            out_sample_results.append({
                "fold": fold_idx,
                "profit_factor": out_result.profit_factor,
                "win_rate": out_result.win_rate,
                "total_trades": out_result.total_trades,
            })

            if show_progress:
                logger.info(f"  In-sample PF: {in_result.profit_factor:.2f}")
                logger.info(f"  Out-sample PF: {out_result.profit_factor:.2f}")

        # Calculate averages
        avg_in_pf = sum(r["profit_factor"] for r in in_sample_results) / len(in_sample_results)
        avg_out_pf = sum(r["profit_factor"] for r in out_sample_results) / len(out_sample_results)

        # Calculate robustness ratio
        robustness = avg_out_pf / avg_in_pf if avg_in_pf > 0 else 0

        # Determine if robust
        is_robust = robustness > 0.5 and avg_out_pf > 1.0

        if show_progress:
            logger.info(f"=== Summary ===")
            logger.info(f"  Avg In-sample PF: {avg_in_pf:.2f}")
            logger.info(f"  Avg Out-sample PF: {avg_out_pf:.2f}")
            logger.info(f"  Robustness Ratio: {robustness:.2f}")
            logger.info(f"  Is Robust: {is_robust}")

        return WalkForwardResult(
            in_sample_results=in_sample_results,
            out_sample_results=out_sample_results,
            avg_in_sample_pf=avg_in_pf,
            avg_out_sample_pf=avg_out_pf,
            robustness_ratio=robustness,
            is_robust=is_robust,
        )


def run_walk_forward(
    prices: List[Dict],
    backtest_fn: Callable,
    initial_params: Dict[str, Any],
    n_splits: int = 5,
    optuna_trials: int = 30,
    seed: int = None,
) -> WalkForwardResult:
    """
    Convenience function to run walk-forward optimization.

    Args:
        prices: List of OHLC price dicts
        backtest_fn: Function(prices, params) -> BacktestResult
        initial_params: Starting parameters
        n_splits: Number of walk-forward folds
        optuna_trials: Optuna trials per fold
        seed: Random seed

    Returns:
        WalkForwardResult with robustness metrics
    """
    optimizer = WalkForwardOptimizer(
        n_splits=n_splits,
        optuna_trials=optuna_trials,
        seed=seed,
    )
    return optimizer.run(prices, backtest_fn, initial_params)
