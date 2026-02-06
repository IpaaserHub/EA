"""
Tests for Optimizer Scheduler Module
====================================
Run with: ./venv/bin/python -m pytest tests/test_scheduler.py -v

Tests for:
1. OptimizerScheduler initialization
2. Manual optimization runs
3. Log management
4. Scheduler start/stop
"""

import pytest
import os
import sys
import tempfile
import json
import shutil
from unittest.mock import Mock, MagicMock
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scheduler.optimizer_scheduler import (
    OptimizerScheduler,
    create_scheduler,
)


# ==================== Test Fixtures ====================

@pytest.fixture
def temp_dirs():
    """Create temporary directories."""
    temp_base = tempfile.mkdtemp()
    data_dir = os.path.join(temp_base, "data")
    config_dir = os.path.join(temp_base, "config")
    log_dir = os.path.join(temp_base, "logs")
    os.makedirs(data_dir)
    os.makedirs(config_dir)
    os.makedirs(log_dir)
    yield data_dir, config_dir, log_dir
    shutil.rmtree(temp_base)


@pytest.fixture
def mock_optimization_loop():
    """Create a mock optimization loop."""
    mock_loop = Mock()
    mock_loop.config_dir = "config/params"

    # Mock optimize_symbol to return a mock run
    mock_run = Mock()
    mock_run.improvement_pct = 15.5
    mock_run.applied = True
    mock_run.reason = "Applied automatically"
    mock_run.old_result = Mock()
    mock_run.old_result.profit_factor = 1.0
    mock_run.new_result = Mock()
    mock_run.new_result.profit_factor = 1.15

    mock_loop.optimize_symbol = Mock(return_value=mock_run)

    return mock_loop


@pytest.fixture
def sample_log_data():
    """Sample log data for testing."""
    return {
        "timestamp": "2025-01-01T02:00:00",
        "n_trials": 50,
        "auto_apply": False,
        "results": [
            {
                "symbol": "XAUJPY",
                "status": "success",
                "improvement_pct": 12.5,
                "applied": True,
                "reason": "Applied automatically",
                "old_profit_factor": 1.0,
                "new_profit_factor": 1.125,
            }
        ],
    }


# ==================== Initialization Tests ====================

class TestSchedulerInit:
    """Tests for OptimizerScheduler initialization."""

    def test_init_with_defaults(self, mock_optimization_loop, temp_dirs):
        """Should initialize with default values."""
        _, _, log_dir = temp_dirs
        scheduler = OptimizerScheduler(
            optimization_loop=mock_optimization_loop,
            log_dir=log_dir,
        )

        assert scheduler.optimization_loop == mock_optimization_loop
        assert scheduler.symbols is None
        assert scheduler.schedule_hour == 2
        assert scheduler.schedule_minute == 0
        assert scheduler.n_trials == 50
        assert scheduler.auto_apply is False

    def test_init_with_custom_values(self, mock_optimization_loop, temp_dirs):
        """Should accept custom values."""
        _, _, log_dir = temp_dirs
        scheduler = OptimizerScheduler(
            optimization_loop=mock_optimization_loop,
            symbols=["XAUJPY", "BTCJPY"],
            log_dir=log_dir,
            schedule_hour=3,
            schedule_minute=30,
            n_trials=100,
            auto_apply=True,
        )

        assert scheduler.symbols == ["XAUJPY", "BTCJPY"]
        assert scheduler.schedule_hour == 3
        assert scheduler.schedule_minute == 30
        assert scheduler.n_trials == 100
        assert scheduler.auto_apply is True

    def test_init_creates_log_dir(self, mock_optimization_loop, temp_dirs):
        """Should create log directory if it doesn't exist."""
        data_dir, _, _ = temp_dirs
        new_log_dir = os.path.join(data_dir, "new_logs")

        scheduler = OptimizerScheduler(
            optimization_loop=mock_optimization_loop,
            log_dir=new_log_dir,
        )

        assert os.path.exists(new_log_dir)

    def test_is_running_initially_false(self, mock_optimization_loop, temp_dirs):
        """Scheduler should not be running initially."""
        _, _, log_dir = temp_dirs
        scheduler = OptimizerScheduler(
            optimization_loop=mock_optimization_loop,
            log_dir=log_dir,
        )

        assert scheduler.is_running() is False


# ==================== Manual Run Tests ====================

class TestManualRun:
    """Tests for manual optimization runs."""

    def test_run_now_calls_optimize(self, mock_optimization_loop, temp_dirs):
        """run_now() should call optimization loop."""
        _, _, log_dir = temp_dirs
        scheduler = OptimizerScheduler(
            optimization_loop=mock_optimization_loop,
            symbols=["XAUJPY"],
            log_dir=log_dir,
        )

        results = scheduler.run_now()

        mock_optimization_loop.optimize_symbol.assert_called_once()
        assert len(results) == 1

    def test_run_now_returns_results(self, mock_optimization_loop, temp_dirs):
        """run_now() should return results list."""
        _, _, log_dir = temp_dirs
        scheduler = OptimizerScheduler(
            optimization_loop=mock_optimization_loop,
            symbols=["XAUJPY"],
            log_dir=log_dir,
        )

        results = scheduler.run_now()

        assert isinstance(results, list)
        assert results[0]["symbol"] == "XAUJPY"
        assert results[0]["status"] == "success"

    def test_run_now_handles_error(self, mock_optimization_loop, temp_dirs):
        """run_now() should handle optimization errors."""
        _, _, log_dir = temp_dirs
        mock_optimization_loop.optimize_symbol.side_effect = ValueError("Test error")

        scheduler = OptimizerScheduler(
            optimization_loop=mock_optimization_loop,
            symbols=["XAUJPY"],
            log_dir=log_dir,
        )

        results = scheduler.run_now()

        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "Test error" in results[0]["error"]

    def test_run_now_multiple_symbols(self, mock_optimization_loop, temp_dirs):
        """run_now() should optimize all symbols."""
        _, _, log_dir = temp_dirs
        scheduler = OptimizerScheduler(
            optimization_loop=mock_optimization_loop,
            symbols=["XAUJPY", "BTCJPY", "USDJPY"],
            log_dir=log_dir,
        )

        results = scheduler.run_now()

        assert len(results) == 3
        assert mock_optimization_loop.optimize_symbol.call_count == 3


# ==================== Log Management Tests ====================

class TestLogManagement:
    """Tests for log file management."""

    def test_run_creates_log_file(self, mock_optimization_loop, temp_dirs):
        """Running optimization should create a log file."""
        _, _, log_dir = temp_dirs
        scheduler = OptimizerScheduler(
            optimization_loop=mock_optimization_loop,
            symbols=["XAUJPY"],
            log_dir=log_dir,
        )

        scheduler.run_now()

        log_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
        assert len(log_files) == 1

    def test_log_file_contains_results(self, mock_optimization_loop, temp_dirs):
        """Log file should contain optimization results."""
        _, _, log_dir = temp_dirs
        scheduler = OptimizerScheduler(
            optimization_loop=mock_optimization_loop,
            symbols=["XAUJPY"],
            log_dir=log_dir,
        )

        scheduler.run_now()

        log_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
        with open(os.path.join(log_dir, log_files[0]), 'r') as f:
            log_data = json.load(f)

        assert "timestamp" in log_data
        assert "results" in log_data
        assert len(log_data["results"]) == 1

    def test_get_recent_logs_empty(self, mock_optimization_loop, temp_dirs):
        """get_recent_logs() should return empty list if no logs."""
        _, _, log_dir = temp_dirs
        scheduler = OptimizerScheduler(
            optimization_loop=mock_optimization_loop,
            log_dir=log_dir,
        )

        logs = scheduler.get_recent_logs()
        assert logs == []

    def test_get_recent_logs_returns_logs(self, mock_optimization_loop, temp_dirs, sample_log_data):
        """get_recent_logs() should return existing logs."""
        _, _, log_dir = temp_dirs

        # Create a log file
        log_file = os.path.join(log_dir, "optimization_20250101_020000.json")
        with open(log_file, 'w') as f:
            json.dump(sample_log_data, f)

        scheduler = OptimizerScheduler(
            optimization_loop=mock_optimization_loop,
            log_dir=log_dir,
        )

        logs = scheduler.get_recent_logs()

        assert len(logs) == 1
        assert logs[0]["timestamp"] == sample_log_data["timestamp"]

    def test_get_recent_logs_respects_limit(self, mock_optimization_loop, temp_dirs, sample_log_data):
        """get_recent_logs() should respect limit parameter."""
        _, _, log_dir = temp_dirs

        # Create multiple log files
        for i in range(5):
            log_file = os.path.join(log_dir, f"optimization_2025010{i}_020000.json")
            with open(log_file, 'w') as f:
                json.dump(sample_log_data, f)

        scheduler = OptimizerScheduler(
            optimization_loop=mock_optimization_loop,
            log_dir=log_dir,
        )

        logs = scheduler.get_recent_logs(limit=3)
        assert len(logs) == 3


# ==================== Scheduler Start/Stop Tests ====================

class TestSchedulerStartStop:
    """Tests for scheduler start and stop functionality."""

    def test_start_returns_true(self, mock_optimization_loop, temp_dirs):
        """start() should return True on success."""
        _, _, log_dir = temp_dirs
        scheduler = OptimizerScheduler(
            optimization_loop=mock_optimization_loop,
            log_dir=log_dir,
        )

        try:
            result = scheduler.start()
            assert result is True
            assert scheduler.is_running() is True
        finally:
            scheduler.stop()

    def test_stop_returns_true(self, mock_optimization_loop, temp_dirs):
        """stop() should return True on success."""
        _, _, log_dir = temp_dirs
        scheduler = OptimizerScheduler(
            optimization_loop=mock_optimization_loop,
            log_dir=log_dir,
        )

        scheduler.start()
        result = scheduler.stop()

        assert result is True
        assert scheduler.is_running() is False

    def test_start_twice_returns_false(self, mock_optimization_loop, temp_dirs):
        """Starting twice should return False."""
        _, _, log_dir = temp_dirs
        scheduler = OptimizerScheduler(
            optimization_loop=mock_optimization_loop,
            log_dir=log_dir,
        )

        try:
            scheduler.start()
            result = scheduler.start()
            assert result is False
        finally:
            scheduler.stop()

    def test_stop_without_start_returns_false(self, mock_optimization_loop, temp_dirs):
        """Stopping without starting should return False."""
        _, _, log_dir = temp_dirs
        scheduler = OptimizerScheduler(
            optimization_loop=mock_optimization_loop,
            log_dir=log_dir,
        )

        result = scheduler.stop()
        assert result is False

    def test_get_next_run_time_when_running(self, mock_optimization_loop, temp_dirs):
        """get_next_run_time() should return datetime when running."""
        _, _, log_dir = temp_dirs
        scheduler = OptimizerScheduler(
            optimization_loop=mock_optimization_loop,
            log_dir=log_dir,
        )

        try:
            scheduler.start()
            next_run = scheduler.get_next_run_time()
            assert next_run is not None
            assert isinstance(next_run, datetime)
        finally:
            scheduler.stop()

    def test_get_next_run_time_when_not_running(self, mock_optimization_loop, temp_dirs):
        """get_next_run_time() should return None when not running."""
        _, _, log_dir = temp_dirs
        scheduler = OptimizerScheduler(
            optimization_loop=mock_optimization_loop,
            log_dir=log_dir,
        )

        next_run = scheduler.get_next_run_time()
        assert next_run is None


# ==================== Convenience Function Tests ====================

class TestCreateScheduler:
    """Tests for create_scheduler convenience function."""

    def test_create_scheduler_returns_scheduler(self, temp_dirs):
        """create_scheduler() should return OptimizerScheduler."""
        data_dir, config_dir, log_dir = temp_dirs

        scheduler = create_scheduler(
            data_dir=data_dir,
            config_dir=config_dir,
            log_dir=log_dir,
            use_ai=False,
        )

        assert isinstance(scheduler, OptimizerScheduler)

    def test_create_scheduler_with_symbols(self, temp_dirs):
        """create_scheduler() should accept symbols parameter."""
        data_dir, config_dir, log_dir = temp_dirs

        scheduler = create_scheduler(
            data_dir=data_dir,
            config_dir=config_dir,
            log_dir=log_dir,
            symbols=["XAUJPY"],
            use_ai=False,
        )

        assert scheduler.symbols == ["XAUJPY"]


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
