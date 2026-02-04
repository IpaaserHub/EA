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
from typing import List, Dict, Any
from dataclasses import dataclass

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
