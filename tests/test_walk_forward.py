"""
Tests for Walk-Forward Optimization Module
==========================================
Run with: pytest tests/test_walk_forward.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimizer.walk_forward import WalkForwardResult, WalkForwardOptimizer


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
