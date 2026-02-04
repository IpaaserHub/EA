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
