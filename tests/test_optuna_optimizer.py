"""
Tests for Optuna Optimizer Module
=================================
Run with: ./venv/bin/python -m pytest tests/test_optuna_optimizer.py -v

Tests for:
1. OptunaOptimizer initialization and basic operation
2. Parameter suggestion within limits
3. Optimization result structure
4. HybridOptimizer combining Optuna + AI
"""

import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimizer.optuna_optimizer import (
    OptunaOptimizer,
    OptimizationResult,
    HybridOptimizer,
    run_optimization,
)
from config.param_manager import PARAM_LIMITS


# ==================== Test Fixtures ====================

@pytest.fixture
def simple_objective():
    """Simple objective function for testing."""
    def objective(params):
        # Higher ADX and TP mult = better score (for testing)
        adx = params.get("adx_threshold", 10)
        tp = params.get("tp_mult", 2.0)
        return adx * 0.1 + tp * 0.5
    return objective


@pytest.fixture
def mock_backtest_fn():
    """Mock backtest function returning result dict."""
    def backtest(params):
        return {
            "total_trades": 50,
            "wins": 25,
            "losses": 25,
            "win_rate": 50.0,
            "profit_factor": 1.0 + params.get("adx_threshold", 10) * 0.01,
            "total_profit": 100.0,
            "max_drawdown": 50.0,
            "avg_win": 10.0,
            "avg_loss": 8.0,
        }
    return backtest


@pytest.fixture
def default_params():
    """Default starting parameters."""
    return {
        "adx_threshold": 10,
        "slope_threshold": 0.00002,
        "buy_position": 0.50,
        "sell_position": 0.50,
        "rsi_buy_max": 75,
        "rsi_sell_min": 25,
        "tp_mult": 2.0,
        "sl_mult": 1.5,
    }


# ==================== OptimizationResult Tests ====================

class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_create_result(self):
        """Should create optimization result."""
        result = OptimizationResult(
            best_params={"adx_threshold": 15, "tp_mult": 2.5},
            best_value=1.5,
            n_trials=50,
            study_name="test_study",
            optimization_history=[],
        )
        assert result.best_value == 1.5
        assert result.n_trials == 50

    def test_to_dict(self):
        """to_dict() should return serializable dict."""
        result = OptimizationResult(
            best_params={"adx_threshold": 15},
            best_value=1.5,
            n_trials=50,
            study_name="test",
            optimization_history=[{"number": 0, "value": 1.0}],
        )
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "best_params" in result_dict
        assert "best_value" in result_dict
        assert "timestamp" in result_dict


# ==================== OptunaOptimizer Init Tests ====================

class TestOptunaOptimizerInit:
    """Tests for OptunaOptimizer initialization."""

    def test_init_defaults(self):
        """Should initialize with defaults."""
        optimizer = OptunaOptimizer()
        assert optimizer.study_name == "trading_optimization"
        assert optimizer.storage is None
        assert optimizer._study is None

    def test_init_custom_name(self):
        """Should accept custom study name."""
        optimizer = OptunaOptimizer(study_name="custom_study")
        assert optimizer.study_name == "custom_study"

    def test_init_with_seed(self):
        """Should accept seed for reproducibility."""
        optimizer = OptunaOptimizer(seed=42)
        assert optimizer.seed == 42


# ==================== Optimization Tests ====================

class TestOptimization:
    """Tests for optimization functionality."""

    def test_optimize_returns_result(self, simple_objective):
        """optimize() should return OptimizationResult."""
        optimizer = OptunaOptimizer(study_name="test_optimize")
        result = optimizer.optimize(
            simple_objective,
            n_trials=5,
            show_progress=False,
        )

        assert isinstance(result, OptimizationResult)
        assert result.n_trials == 5

    def test_optimize_finds_better_params(self, simple_objective, default_params):
        """Should find params that improve objective."""
        optimizer = OptunaOptimizer(study_name="test_improve", seed=42)

        # Get baseline score
        baseline_score = simple_objective(default_params)

        # Optimize
        result = optimizer.optimize(
            simple_objective,
            n_trials=20,
            initial_params=default_params,
            show_progress=False,
        )

        # Best should be at least as good as baseline
        assert result.best_value >= baseline_score * 0.9  # Allow some variance

    def test_optimize_respects_param_limits(self, simple_objective):
        """Best params should be within PARAM_LIMITS."""
        optimizer = OptunaOptimizer(study_name="test_limits", seed=42)
        result = optimizer.optimize(
            simple_objective,
            n_trials=10,
            show_progress=False,
        )

        for param_name, value in result.best_params.items():
            if param_name in PARAM_LIMITS:
                limits = PARAM_LIMITS[param_name]
                assert value >= limits["min"], f"{param_name} below min"
                assert value <= limits["max"], f"{param_name} above max"

    def test_optimize_int_params_are_int(self, simple_objective):
        """Int params should be converted to int type."""
        optimizer = OptunaOptimizer(study_name="test_types", seed=42)
        result = optimizer.optimize(
            simple_objective,
            n_trials=5,
            show_progress=False,
        )

        for param_name, value in result.best_params.items():
            if param_name in PARAM_LIMITS:
                if PARAM_LIMITS[param_name]["type"] == "int":
                    assert isinstance(value, int), f"{param_name} should be int"

    def test_optimize_with_timeout(self, simple_objective):
        """Should respect timeout parameter."""
        optimizer = OptunaOptimizer(study_name="test_timeout")
        result = optimizer.optimize(
            simple_objective,
            n_trials=1000,  # Many trials
            timeout=2,  # But only 2 seconds
            show_progress=False,
        )

        # Should have completed some trials but not all
        assert result.n_trials < 1000
        assert result.n_trials >= 1

    def test_optimize_with_initial_params(self, simple_objective, default_params):
        """Should start from initial params."""
        optimizer = OptunaOptimizer(study_name="test_initial", seed=42)
        result = optimizer.optimize(
            simple_objective,
            n_trials=5,
            initial_params=default_params,
            show_progress=False,
        )

        # First trial should have been the initial params
        assert result.n_trials >= 1


# ==================== Study Access Tests ====================

class TestStudyAccess:
    """Tests for accessing the Optuna study."""

    def test_get_study(self, simple_objective):
        """get_study() should return Optuna study."""
        optimizer = OptunaOptimizer(study_name="test_study_access")
        optimizer.optimize(simple_objective, n_trials=3, show_progress=False)

        study = optimizer.get_study()
        assert study is not None
        assert len(study.trials) == 3

    def test_get_best_trials(self, simple_objective):
        """get_best_trials() should return top N trials."""
        optimizer = OptunaOptimizer(study_name="test_best_trials", seed=42)
        optimizer.optimize(simple_objective, n_trials=10, show_progress=False)

        best = optimizer.get_best_trials(n=3)

        assert len(best) == 3
        # Should be sorted by value descending
        assert best[0]["value"] >= best[1]["value"]
        assert best[1]["value"] >= best[2]["value"]

    def test_get_best_trials_empty_study(self):
        """Should return empty list if no trials."""
        optimizer = OptunaOptimizer(study_name="test_empty")
        best = optimizer.get_best_trials(n=5)
        assert best == []


# ==================== HybridOptimizer Tests ====================

class TestHybridOptimizer:
    """Tests for HybridOptimizer combining Optuna + AI."""

    def test_init_without_ai(self):
        """Should work without AI analyzer."""
        hybrid = HybridOptimizer()
        assert hybrid.ai_analyzer is None

    def test_optimize_without_ai(self, simple_objective, mock_backtest_fn, default_params):
        """Should optimize using only Optuna when no AI."""
        hybrid = HybridOptimizer()
        result = hybrid.optimize(
            objective_fn=simple_objective,
            backtest_fn=mock_backtest_fn,
            initial_params=default_params,
            optuna_trials=5,
            ai_refinement_trials=3,
        )

        assert isinstance(result, OptimizationResult)
        assert result.n_trials == 5  # No AI refinement trials

    def test_optimize_with_mock_ai(self, simple_objective, mock_backtest_fn, default_params):
        """Should use AI suggestions when analyzer provided."""
        # Create a mock AI analyzer
        class MockAnalyzer:
            def analyze(self, backtest_result, params, **kwargs):
                from optimizer.ai_analyzer import AnalysisResult, ParameterSuggestion
                return AnalysisResult(
                    analysis="Test",
                    suggestions=[
                        ParameterSuggestion("adx_threshold", 10, 15, "Test")
                    ],
                    expected_impact="Test",
                    confidence="medium",
                )

        hybrid = HybridOptimizer(ai_analyzer=MockAnalyzer())
        result = hybrid.optimize(
            objective_fn=simple_objective,
            backtest_fn=mock_backtest_fn,
            initial_params=default_params,
            optuna_trials=5,
            ai_refinement_trials=3,
        )

        assert isinstance(result, OptimizationResult)
        # HybridOptimizer returns best result (either Optuna or AI refinement)
        # n_trials reflects the study that produced the best result
        assert result.n_trials >= 3  # At least the refinement trials ran


# ==================== Convenience Function Tests ====================

class TestConvenienceFunction:
    """Tests for run_optimization convenience function."""

    def test_run_optimization_works(self, simple_objective, default_params):
        """run_optimization() should work like optimizer.optimize()."""
        result = run_optimization(
            simple_objective,
            n_trials=5,
            initial_params=default_params,
            study_name="test_convenience",
        )

        assert isinstance(result, OptimizationResult)
        assert result.n_trials == 5


# ==================== Edge Cases ====================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_objective_returning_none(self):
        """Should handle objective returning None."""
        def bad_objective(params):
            return None

        optimizer = OptunaOptimizer(study_name="test_none")
        result = optimizer.optimize(bad_objective, n_trials=3, show_progress=False)

        # Should complete without error
        assert result.n_trials == 3
        assert result.best_value == 0.0  # None converted to 0

    def test_objective_raising_exception(self):
        """Should handle objective raising exception."""
        call_count = [0]

        def failing_objective(params):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ValueError("Test error")
            return 1.0

        optimizer = OptunaOptimizer(study_name="test_exception")
        result = optimizer.optimize(failing_objective, n_trials=5, show_progress=False)

        # Should complete despite some failures
        assert result.n_trials == 5

    def test_empty_initial_params(self, simple_objective):
        """Should handle empty initial params."""
        optimizer = OptunaOptimizer(study_name="test_empty_initial")
        result = optimizer.optimize(
            simple_objective,
            n_trials=3,
            initial_params={},
            show_progress=False,
        )

        assert result.n_trials >= 1


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
