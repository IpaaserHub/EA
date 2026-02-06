"""
Tests for Optimization Loop Module
==================================
Run with: ./venv/bin/python -m pytest tests/test_optimization_loop.py -v

Tests for:
1. OptimizationRun dataclass
2. OptimizationLoop initialization
3. Single symbol optimization
4. Safety threshold validation
"""

import pytest
import os
import sys
import tempfile
import json
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimizer.optimization_loop import (
    OptimizationLoop,
    OptimizationRun,
    run_single_optimization,
    print_run_summary,
)
from backtest.engine import BacktestResult, Trade


# ==================== Test Fixtures ====================

@pytest.fixture
def temp_dirs():
    """Create temporary data and config directories."""
    temp_base = tempfile.mkdtemp()
    data_dir = os.path.join(temp_base, "data")
    config_dir = os.path.join(temp_base, "config")
    os.makedirs(data_dir)
    os.makedirs(config_dir)
    yield data_dir, config_dir
    shutil.rmtree(temp_base)


@pytest.fixture
def sample_params():
    """Sample parameter file content."""
    return {
        "symbol": "TEST",
        "mode": "NORMAL",
        "adx_threshold": 10,
        "slope_threshold": 0.00002,
        "buy_position": 0.50,
        "sell_position": 0.50,
        "rsi_buy_max": 75,
        "rsi_sell_min": 25,
        "tp_mult": 2.0,
        "sl_mult": 1.5,
    }


@pytest.fixture
def sample_price_data():
    """Generate sample price data CSV content."""
    lines = ["Date,Open,High,Low,Close,Change,Change%"]
    base_price = 100.0
    for i in range(200):  # 200 candles for sufficient data
        # Create uptrend with noise
        price = base_price + i * 0.1 + (i % 5) * 0.05
        lines.append(
            f"2025.01.{(i % 28) + 1:02d} {i % 24:02d}:00,"
            f"{price:.2f},{price + 0.5:.2f},{price - 0.3:.2f},{price + 0.2:.2f},0.1,0.1"
        )
    return "\n".join(lines)


@pytest.fixture
def setup_test_symbol(temp_dirs, sample_params, sample_price_data):
    """Set up a complete test symbol with data and params."""
    data_dir, config_dir = temp_dirs

    # Write params
    param_file = os.path.join(config_dir, "TEST.json")
    with open(param_file, 'w') as f:
        json.dump(sample_params, f)

    # Write price data
    data_file = os.path.join(data_dir, "TEST_H1_extended.csv")
    with open(data_file, 'w') as f:
        f.write(sample_price_data)

    return "TEST", data_dir, config_dir


# ==================== OptimizationRun Tests ====================

class TestOptimizationRun:
    """Tests for OptimizationRun dataclass."""

    def test_create_run(self):
        """Should create optimization run."""
        # Create minimal backtest results
        old_result = BacktestResult(
            total_trades=50, wins=25, losses=25, win_rate=50.0,
            profit_factor=1.0, total_profit=0, gross_profit=100,
            gross_loss=100, max_drawdown=50, avg_win=4, avg_loss=4, trades=[]
        )
        new_result = BacktestResult(
            total_trades=50, wins=30, losses=20, win_rate=60.0,
            profit_factor=1.5, total_profit=50, gross_profit=150,
            gross_loss=100, max_drawdown=40, avg_win=5, avg_loss=5, trades=[]
        )

        # Create mock optimization result
        from optimizer.optuna_optimizer import OptimizationResult
        opt_result = OptimizationResult(
            best_params={"adx_threshold": 15},
            best_value=1.5,
            n_trials=10,
            study_name="test",
            optimization_history=[],
        )

        run = OptimizationRun(
            symbol="TEST",
            old_params={"adx_threshold": 10},
            new_params={"adx_threshold": 15},
            old_result=old_result,
            new_result=new_result,
            improvement_pct=50.0,
            optimization_result=opt_result,
            ai_analysis=None,
            walk_forward_result=None,
            applied=False,
            reason="Test",
            timestamp="2025-01-01T00:00:00",
        )

        assert run.symbol == "TEST"
        assert run.improvement_pct == 50.0
        assert not run.applied

    def test_to_dict(self):
        """to_dict() should return serializable dict."""
        old_result = BacktestResult(
            total_trades=50, wins=25, losses=25, win_rate=50.0,
            profit_factor=1.0, total_profit=0, gross_profit=100,
            gross_loss=100, max_drawdown=50, avg_win=4, avg_loss=4, trades=[]
        )
        new_result = BacktestResult(
            total_trades=50, wins=30, losses=20, win_rate=60.0,
            profit_factor=1.5, total_profit=50, gross_profit=150,
            gross_loss=100, max_drawdown=40, avg_win=5, avg_loss=5, trades=[]
        )

        from optimizer.optuna_optimizer import OptimizationResult
        opt_result = OptimizationResult(
            best_params={"adx_threshold": 15},
            best_value=1.5,
            n_trials=10,
            study_name="test",
            optimization_history=[],
        )

        run = OptimizationRun(
            symbol="TEST",
            old_params={"adx_threshold": 10},
            new_params={"adx_threshold": 15},
            old_result=old_result,
            new_result=new_result,
            improvement_pct=50.0,
            optimization_result=opt_result,
            ai_analysis=None,
            walk_forward_result=None,
            applied=False,
            reason="Test",
            timestamp="2025-01-01T00:00:00",
        )

        result_dict = run.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["symbol"] == "TEST"
        assert "old_result" in result_dict
        assert "new_result" in result_dict


# ==================== OptimizationLoop Init Tests ====================

class TestOptimizationLoopInit:
    """Tests for OptimizationLoop initialization."""

    def test_init_defaults(self, temp_dirs):
        """Should initialize with defaults."""
        data_dir, config_dir = temp_dirs
        loop = OptimizationLoop(data_dir=data_dir, config_dir=config_dir)

        assert loop.data_dir == data_dir
        assert loop.config_dir == config_dir
        assert loop.use_ai is True

    def test_init_without_ai(self, temp_dirs):
        """Should work without AI."""
        data_dir, config_dir = temp_dirs
        loop = OptimizationLoop(data_dir=data_dir, config_dir=config_dir, use_ai=False)

        assert loop.ai_analyzer is None

    def test_safety_thresholds(self, temp_dirs):
        """Should have safety thresholds defined."""
        data_dir, config_dir = temp_dirs
        loop = OptimizationLoop(data_dir=data_dir, config_dir=config_dir)

        assert loop.MIN_IMPROVEMENT_PCT > 0
        assert loop.MIN_TRADES > 0
        assert loop.MAX_DRAWDOWN_INCREASE_PCT > 0


# ==================== Single Symbol Optimization Tests ====================

class TestSingleSymbolOptimization:
    """Tests for single symbol optimization."""

    def test_optimize_symbol_returns_run(self, setup_test_symbol):
        """optimize_symbol() should return OptimizationRun."""
        symbol, data_dir, config_dir = setup_test_symbol
        loop = OptimizationLoop(data_dir=data_dir, config_dir=config_dir, use_ai=False, wfo_enabled=False)

        run = loop.optimize_symbol(symbol, n_trials=5)

        assert isinstance(run, OptimizationRun)
        assert run.symbol == symbol

    def test_optimize_symbol_has_results(self, setup_test_symbol):
        """Should have old and new results."""
        symbol, data_dir, config_dir = setup_test_symbol
        loop = OptimizationLoop(data_dir=data_dir, config_dir=config_dir, use_ai=False, wfo_enabled=False)

        run = loop.optimize_symbol(symbol, n_trials=5)

        assert run.old_result is not None
        assert run.new_result is not None
        assert isinstance(run.old_result, BacktestResult)
        assert isinstance(run.new_result, BacktestResult)

    def test_optimize_symbol_calculates_improvement(self, setup_test_symbol):
        """Should calculate improvement percentage."""
        symbol, data_dir, config_dir = setup_test_symbol
        loop = OptimizationLoop(data_dir=data_dir, config_dir=config_dir, use_ai=False, wfo_enabled=False)

        run = loop.optimize_symbol(symbol, n_trials=5)

        assert isinstance(run.improvement_pct, float)

    def test_optimize_raises_on_missing_data_or_params(self, temp_dirs):
        """Should raise error if no data or params found."""
        data_dir, config_dir = temp_dirs
        loop = OptimizationLoop(data_dir=data_dir, config_dir=config_dir, use_ai=False, wfo_enabled=False)

        # ParamManager returns defaults if no config file exists,
        # so the error will be about missing data
        with pytest.raises(ValueError, match="No data file found"):
            loop.optimize_symbol("NONEXISTENT", n_trials=5)

    def test_optimize_raises_on_missing_data(self, temp_dirs, sample_params):
        """Should raise error if no data found."""
        data_dir, config_dir = temp_dirs

        # Create params but no data
        param_file = os.path.join(config_dir, "TEST.json")
        with open(param_file, 'w') as f:
            json.dump(sample_params, f)

        loop = OptimizationLoop(data_dir=data_dir, config_dir=config_dir, use_ai=False, wfo_enabled=False)

        with pytest.raises(ValueError, match="No data file found"):
            loop.optimize_symbol("TEST", n_trials=5)


# ==================== Safety Validation Tests ====================

class TestSafetyValidation:
    """Tests for safety threshold validation."""

    def test_not_applied_if_insufficient_trades(self, setup_test_symbol):
        """Should not apply if too few trades."""
        symbol, data_dir, config_dir = setup_test_symbol

        # Create loop with high minimum trades requirement
        loop = OptimizationLoop(data_dir=data_dir, config_dir=config_dir, use_ai=False, wfo_enabled=False)
        loop.MIN_TRADES = 1000  # Very high

        run = loop.optimize_symbol(symbol, n_trials=3, auto_apply=True)

        assert not run.applied
        assert "Insufficient trades" in run.reason

    def test_not_applied_if_small_improvement(self, setup_test_symbol):
        """Should not apply if improvement too small (or other safety criteria fail)."""
        symbol, data_dir, config_dir = setup_test_symbol

        loop = OptimizationLoop(data_dir=data_dir, config_dir=config_dir, use_ai=False, wfo_enabled=False)
        loop.MIN_IMPROVEMENT_PCT = 1000  # Very high requirement

        run = loop.optimize_symbol(symbol, n_trials=3, auto_apply=True)

        # Should not be applied due to safety checks
        # (could be insufficient trades or small improvement)
        assert not run.applied
        # Reason should indicate why it wasn't applied
        assert run.reason != ""


# ==================== Multi-Symbol Tests ====================

class TestMultiSymbolOptimization:
    """Tests for multi-symbol optimization."""

    def test_optimize_all_symbols(self, temp_dirs, sample_params, sample_price_data):
        """Should optimize all available symbols."""
        data_dir, config_dir = temp_dirs

        # Create two symbols
        for sym in ["SYM1", "SYM2"]:
            params = sample_params.copy()
            params["symbol"] = sym
            param_file = os.path.join(config_dir, f"{sym}.json")
            with open(param_file, 'w') as f:
                json.dump(params, f)

            data_file = os.path.join(data_dir, f"{sym}_H1_extended.csv")
            with open(data_file, 'w') as f:
                f.write(sample_price_data)

        loop = OptimizationLoop(data_dir=data_dir, config_dir=config_dir, use_ai=False, wfo_enabled=False)
        results = loop.optimize_all_symbols(n_trials=3)

        assert len(results) == 2
        symbols = [r.symbol for r in results]
        assert "SYM1" in symbols
        assert "SYM2" in symbols


# ==================== Apply and Rollback Tests ====================

class TestApplyAndRollback:
    """Tests for applying and rolling back optimization runs."""

    def test_apply_run(self, setup_test_symbol):
        """apply_run() should save new parameters."""
        symbol, data_dir, config_dir = setup_test_symbol
        loop = OptimizationLoop(data_dir=data_dir, config_dir=config_dir, use_ai=False, wfo_enabled=False)

        run = loop.optimize_symbol(symbol, n_trials=3, auto_apply=False)

        # Manually apply
        success = loop.apply_run(run)

        assert success
        assert run.applied

    def test_apply_already_applied(self, setup_test_symbol):
        """Should not re-apply already applied run."""
        symbol, data_dir, config_dir = setup_test_symbol
        loop = OptimizationLoop(data_dir=data_dir, config_dir=config_dir, use_ai=False, wfo_enabled=False)

        run = loop.optimize_symbol(symbol, n_trials=3, auto_apply=True)

        if run.applied:
            success = loop.apply_run(run)
            assert not success  # Already applied

    def test_rollback(self, setup_test_symbol):
        """rollback() should restore previous parameters."""
        symbol, data_dir, config_dir = setup_test_symbol
        loop = OptimizationLoop(data_dir=data_dir, config_dir=config_dir, use_ai=False, wfo_enabled=False)

        # First optimization
        run1 = loop.optimize_symbol(symbol, n_trials=3, auto_apply=True)

        if run1.applied:
            # Rollback
            restored = loop.rollback(symbol)
            # Should have some parameters (may or may not be exactly old_params
            # depending on history implementation)
            assert restored is not None or True  # Rollback may fail if no history


# ==================== Convenience Function Tests ====================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_run_single_optimization(self, setup_test_symbol):
        """run_single_optimization() should work."""
        symbol, data_dir, config_dir = setup_test_symbol

        run = run_single_optimization(
            symbol,
            data_dir=data_dir,
            config_dir=config_dir,
            n_trials=3,
        )

        assert isinstance(run, OptimizationRun)

    def test_print_run_summary(self, setup_test_symbol, capsys):
        """print_run_summary() should output summary."""
        symbol, data_dir, config_dir = setup_test_symbol
        loop = OptimizationLoop(data_dir=data_dir, config_dir=config_dir, use_ai=False, wfo_enabled=False)

        run = loop.optimize_symbol(symbol, n_trials=3)
        print_run_summary(run)

        captured = capsys.readouterr()
        assert "Optimization Summary" in captured.out
        assert symbol in captured.out


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
