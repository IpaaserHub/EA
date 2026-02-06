# Walk-Forward Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add walk-forward optimization to prevent overfitting and validate trading strategy robustness.

**Architecture:** Use sklearn's TimeSeriesSplit to divide price data into multiple folds. For each fold, optimize parameters on in-sample data using existing OptunaOptimizer, then validate on out-of-sample data using BacktestEngine. Track robustness ratio (out-of-sample / in-sample performance).

**Tech Stack:** Python, sklearn (TimeSeriesSplit), existing OptunaOptimizer, existing BacktestEngine

---

## Task 1: Create WalkForwardResult Dataclass

**Files:**
- Create: `src/optimizer/walk_forward.py`
- Test: `tests/test_walk_forward.py`

**Step 1: Write the failing test**

Create `tests/test_walk_forward.py`:

```python
"""
Tests for Walk-Forward Optimization Module
==========================================
Run with: pytest tests/test_walk_forward.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimizer.walk_forward import WalkForwardResult


class TestWalkForwardResult:
    """Tests for WalkForwardResult dataclass."""

    def test_create_result(self):
        """Should create walk-forward result."""
        result = WalkForwardResult(
            in_sample_results=[{"fold": 0, "profit_factor": 1.5}],
            out_sample_results=[{"fold": 0, "profit_factor": 1.2}],
            avg_in_sample_pf=1.5,
            avg_out_sample_pf=1.2,
            robustness_ratio=0.8,
            is_robust=True,
        )
        assert result.robustness_ratio == 0.8
        assert result.is_robust is True

    def test_to_dict(self):
        """to_dict() should return serializable dict."""
        result = WalkForwardResult(
            in_sample_results=[{"fold": 0, "profit_factor": 1.5}],
            out_sample_results=[{"fold": 0, "profit_factor": 1.2}],
            avg_in_sample_pf=1.5,
            avg_out_sample_pf=1.2,
            robustness_ratio=0.8,
            is_robust=True,
        )
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "robustness_ratio" in result_dict
        assert "is_robust" in result_dict
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_walk_forward.py::TestWalkForwardResult::test_create_result -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'optimizer.walk_forward'"

**Step 3: Write minimal implementation**

Create `src/optimizer/walk_forward.py`:

```python
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

from typing import List, Dict, Any
from dataclasses import dataclass


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
```

**Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/test_walk_forward.py::TestWalkForwardResult -v`

Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/optimizer/walk_forward.py tests/test_walk_forward.py
git commit -m "feat(optimizer): add WalkForwardResult dataclass"
```

---

## Task 2: Create WalkForwardOptimizer Class (Init)

**Files:**
- Modify: `src/optimizer/walk_forward.py`
- Test: `tests/test_walk_forward.py`

**Step 1: Write the failing test**

Add to `tests/test_walk_forward.py`:

```python
from optimizer.walk_forward import WalkForwardResult, WalkForwardOptimizer


class TestWalkForwardOptimizerInit:
    """Tests for WalkForwardOptimizer initialization."""

    def test_init_defaults(self):
        """Should initialize with default settings."""
        wfo = WalkForwardOptimizer()
        assert wfo.n_splits == 5
        assert wfo.min_train_size == 100
        assert wfo.optuna_trials == 30

    def test_init_custom(self):
        """Should accept custom settings."""
        wfo = WalkForwardOptimizer(
            n_splits=3,
            min_train_size=50,
            optuna_trials=10,
        )
        assert wfo.n_splits == 3
        assert wfo.min_train_size == 50
        assert wfo.optuna_trials == 10
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_walk_forward.py::TestWalkForwardOptimizerInit -v`

Expected: FAIL with "cannot import name 'WalkForwardOptimizer'"

**Step 3: Write minimal implementation**

Add to `src/optimizer/walk_forward.py`:

```python
import logging

logger = logging.getLogger(__name__)


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
```

**Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/test_walk_forward.py::TestWalkForwardOptimizerInit -v`

Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/optimizer/walk_forward.py tests/test_walk_forward.py
git commit -m "feat(optimizer): add WalkForwardOptimizer class init"
```

---

## Task 3: Implement _split_data Method

**Files:**
- Modify: `src/optimizer/walk_forward.py`
- Test: `tests/test_walk_forward.py`

**Step 1: Write the failing test**

Add to `tests/test_walk_forward.py`:

```python
class TestWalkForwardSplitData:
    """Tests for data splitting functionality."""

    def test_split_data_returns_folds(self):
        """Should split data into train/test folds."""
        wfo = WalkForwardOptimizer(n_splits=3, min_train_size=10)

        # Create mock price data (50 candles)
        prices = [{"close": 100 + i} for i in range(50)]

        folds = wfo._split_data(prices)

        assert len(folds) == 3
        for train_data, test_data in folds:
            assert len(train_data) >= 10  # min_train_size
            assert len(test_data) > 0

    def test_split_data_no_overlap(self):
        """Train and test data should not overlap."""
        wfo = WalkForwardOptimizer(n_splits=3, min_train_size=10)
        prices = [{"close": i} for i in range(50)]

        folds = wfo._split_data(prices)

        for train_data, test_data in folds:
            train_values = [p["close"] for p in train_data]
            test_values = [p["close"] for p in test_data]

            # No common values (since we used i as close price)
            assert len(set(train_values) & set(test_values)) == 0

    def test_split_data_test_follows_train(self):
        """Test data should come after train data chronologically."""
        wfo = WalkForwardOptimizer(n_splits=3, min_train_size=10)
        prices = [{"close": i} for i in range(50)]

        folds = wfo._split_data(prices)

        for train_data, test_data in folds:
            max_train = max(p["close"] for p in train_data)
            min_test = min(p["close"] for p in test_data)
            assert min_test > max_train  # Test comes after train
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_walk_forward.py::TestWalkForwardSplitData -v`

Expected: FAIL with "AttributeError: 'WalkForwardOptimizer' object has no attribute '_split_data'"

**Step 3: Write minimal implementation**

Add method to `WalkForwardOptimizer` class in `src/optimizer/walk_forward.py`:

```python
    def _split_data(self, prices: List[Dict]) -> List[tuple]:
        """
        Split price data into train/test folds using TimeSeriesSplit.

        Args:
            prices: List of OHLC dicts

        Returns:
            List of (train_data, test_data) tuples
        """
        from sklearn.model_selection import TimeSeriesSplit

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

        return folds
```

Also add the import at top:

```python
from typing import List, Dict, Any, Callable, Optional
```

**Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/test_walk_forward.py::TestWalkForwardSplitData -v`

Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/optimizer/walk_forward.py tests/test_walk_forward.py
git commit -m "feat(optimizer): add walk-forward data splitting"
```

---

## Task 4: Implement run() Method

**Files:**
- Modify: `src/optimizer/walk_forward.py`
- Test: `tests/test_walk_forward.py`

**Step 1: Write the failing test**

Add to `tests/test_walk_forward.py`:

```python
from backtest.engine import BacktestResult


@pytest.fixture
def mock_prices():
    """Generate mock price data (200 candles)."""
    import random
    random.seed(42)

    prices = []
    price = 100.0
    for i in range(200):
        change = random.uniform(-1, 1)
        price += change
        prices.append({
            "open": price,
            "high": price + abs(change),
            "low": price - abs(change),
            "close": price + change * 0.5,
        })
    return prices


@pytest.fixture
def simple_backtest_fn():
    """Simple backtest function for testing."""
    def backtest(prices, params):
        # Return a mock BacktestResult based on params
        adx = params.get("adx_threshold", 10)
        pf = 1.0 + (adx - 10) * 0.02  # Higher ADX = slightly better PF

        return BacktestResult(
            total_trades=20,
            wins=12,
            losses=8,
            win_rate=60.0,
            profit_factor=max(0.5, pf),
            total_profit=100.0,
            gross_profit=150.0,
            gross_loss=50.0,
            max_drawdown=30.0,
            avg_win=12.5,
            avg_loss=6.25,
            trades=[],
        )
    return backtest


class TestWalkForwardRun:
    """Tests for run() method."""

    def test_run_returns_result(self, mock_prices, simple_backtest_fn):
        """run() should return WalkForwardResult."""
        wfo = WalkForwardOptimizer(n_splits=3, optuna_trials=5)

        result = wfo.run(
            prices=mock_prices,
            backtest_fn=simple_backtest_fn,
            initial_params={"adx_threshold": 10},
        )

        assert isinstance(result, WalkForwardResult)
        assert len(result.in_sample_results) > 0
        assert len(result.out_sample_results) > 0

    def test_run_calculates_robustness(self, mock_prices, simple_backtest_fn):
        """Should calculate robustness ratio."""
        wfo = WalkForwardOptimizer(n_splits=3, optuna_trials=5)

        result = wfo.run(
            prices=mock_prices,
            backtest_fn=simple_backtest_fn,
            initial_params={"adx_threshold": 10},
        )

        assert result.avg_in_sample_pf > 0
        assert result.avg_out_sample_pf > 0
        assert result.robustness_ratio > 0

    def test_run_determines_robustness(self, mock_prices, simple_backtest_fn):
        """Should set is_robust based on criteria."""
        wfo = WalkForwardOptimizer(n_splits=3, optuna_trials=5)

        result = wfo.run(
            prices=mock_prices,
            backtest_fn=simple_backtest_fn,
            initial_params={"adx_threshold": 10},
        )

        # is_robust should be True if ratio > 0.5 and out_pf > 1.0
        expected_robust = (
            result.robustness_ratio > 0.5
            and result.avg_out_sample_pf > 1.0
        )
        assert result.is_robust == expected_robust
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_walk_forward.py::TestWalkForwardRun -v`

Expected: FAIL with "AttributeError: 'WalkForwardOptimizer' object has no attribute 'run'"

**Step 3: Write minimal implementation**

Add to `WalkForwardOptimizer` class in `src/optimizer/walk_forward.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/test_walk_forward.py::TestWalkForwardRun -v`

Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/optimizer/walk_forward.py tests/test_walk_forward.py
git commit -m "feat(optimizer): implement walk-forward run() method"
```

---

## Task 5: Add Convenience Function

**Files:**
- Modify: `src/optimizer/walk_forward.py`
- Test: `tests/test_walk_forward.py`

**Step 1: Write the failing test**

Add to `tests/test_walk_forward.py`:

```python
from optimizer.walk_forward import run_walk_forward


class TestConvenienceFunction:
    """Tests for run_walk_forward convenience function."""

    def test_run_walk_forward_works(self, mock_prices, simple_backtest_fn):
        """run_walk_forward() should work like WalkForwardOptimizer.run()."""
        result = run_walk_forward(
            prices=mock_prices,
            backtest_fn=simple_backtest_fn,
            initial_params={"adx_threshold": 10},
            n_splits=3,
            optuna_trials=5,
        )

        assert isinstance(result, WalkForwardResult)
        assert result.robustness_ratio > 0
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_walk_forward.py::TestConvenienceFunction -v`

Expected: FAIL with "cannot import name 'run_walk_forward'"

**Step 3: Write minimal implementation**

Add to bottom of `src/optimizer/walk_forward.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/test_walk_forward.py::TestConvenienceFunction -v`

Expected: PASS (1 test)

**Step 5: Commit**

```bash
git add src/optimizer/walk_forward.py tests/test_walk_forward.py
git commit -m "feat(optimizer): add run_walk_forward convenience function"
```

---

## Task 6: Add Module Exports

**Files:**
- Modify: `src/optimizer/__init__.py`

**Step 1: Write the failing test**

Add to `tests/test_walk_forward.py`:

```python
class TestModuleExports:
    """Tests for module exports."""

    def test_import_from_optimizer(self):
        """Should be able to import from optimizer package."""
        from optimizer import WalkForwardOptimizer, WalkForwardResult, run_walk_forward

        assert WalkForwardOptimizer is not None
        assert WalkForwardResult is not None
        assert run_walk_forward is not None
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_walk_forward.py::TestModuleExports -v`

Expected: FAIL with "cannot import name 'WalkForwardOptimizer' from 'optimizer'"

**Step 3: Write minimal implementation**

Update `src/optimizer/__init__.py`:

```python
"""
Optimizer Package
=================
Contains optimization modules for trading parameter tuning.
"""

from .optuna_optimizer import OptunaOptimizer, OptimizationResult, HybridOptimizer, run_optimization
from .ai_analyzer import AIAnalyzer, AnalysisResult, analyze_backtest
from .walk_forward import WalkForwardOptimizer, WalkForwardResult, run_walk_forward

__all__ = [
    # Optuna
    "OptunaOptimizer",
    "OptimizationResult",
    "HybridOptimizer",
    "run_optimization",
    # AI
    "AIAnalyzer",
    "AnalysisResult",
    "analyze_backtest",
    # Walk-Forward
    "WalkForwardOptimizer",
    "WalkForwardResult",
    "run_walk_forward",
]
```

**Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/test_walk_forward.py::TestModuleExports -v`

Expected: PASS (1 test)

**Step 5: Commit**

```bash
git add src/optimizer/__init__.py tests/test_walk_forward.py
git commit -m "feat(optimizer): export walk-forward module"
```

---

## Task 7: Run All Tests and Final Verification

**Files:**
- All test files

**Step 1: Run all walk-forward tests**

Run: `./venv/bin/python -m pytest tests/test_walk_forward.py -v`

Expected: All tests PASS

**Step 2: Run all optimizer tests**

Run: `./venv/bin/python -m pytest tests/test_optuna_optimizer.py tests/test_walk_forward.py -v`

Expected: All tests PASS

**Step 3: Verify integration with existing backtest engine**

Create a quick integration test by running:

```bash
./venv/bin/python -c "
from src.optimizer import WalkForwardOptimizer, run_walk_forward
from src.backtest.engine import run_backtest

print('Walk-forward imports OK')
print('BacktestEngine imports OK')
print('Integration check passed!')
"
```

Expected: "Integration check passed!"

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat(optimizer): complete walk-forward optimization implementation

- Add WalkForwardResult dataclass
- Add WalkForwardOptimizer with TimeSeriesSplit
- Add run_walk_forward convenience function
- Add comprehensive tests
- Export from optimizer package"
```

---

## Summary

After completing all tasks, you will have:

```
src/optimizer/
├── __init__.py          (updated with exports)
├── optuna_optimizer.py  (unchanged)
├── ai_analyzer.py       (unchanged)
├── walk_forward.py      (NEW - walk-forward optimization)
└── optimization_loop.py (unchanged)

tests/
├── test_walk_forward.py (NEW - 10+ tests)
└── ... (existing tests unchanged)
```

**Usage Example:**

```python
from src.optimizer import run_walk_forward
from src.backtest.engine import run_backtest

# Load your price data
prices = load_prices("XAUUSD", days=365)

# Define backtest function
def backtest_fn(prices, params):
    return run_backtest(prices, params)

# Run walk-forward analysis
result = run_walk_forward(
    prices=prices,
    backtest_fn=backtest_fn,
    initial_params={"adx_threshold": 10, "tp_mult": 2.0},
    n_splits=5,
    optuna_trials=30,
)

print(f"Robustness Ratio: {result.robustness_ratio:.2f}")
print(f"Is Robust: {result.is_robust}")
```
