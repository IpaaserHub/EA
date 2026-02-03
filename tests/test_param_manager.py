"""
Tests for Parameter Manager
===========================
Run with: pytest tests/test_param_manager.py -v

These tests verify that:
1. Parameters can be loaded and saved
2. Validation clamps values to safe limits
3. History tracking works
4. Rollback works
"""

import pytest
import os
import json
import tempfile
import shutil

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.param_manager import ParamManager, DEFAULT_PARAMS, PARAM_LIMITS


class TestParamManager:
    """Test suite for ParamManager."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for test configs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def manager(self, temp_config_dir):
        """Create a ParamManager with temp directory."""
        return ParamManager(config_dir=temp_config_dir)

    # ==================== Load Tests ====================

    def test_load_returns_defaults_when_no_config(self, manager):
        """When no config exists, should return default parameters."""
        params = manager.load("XAUJPY")
        assert params == DEFAULT_PARAMS

    def test_load_returns_saved_params(self, manager):
        """Should return previously saved parameters."""
        custom_params = {"adx_threshold": 10, "tp_mult": 3.0}
        manager.save("XAUJPY", custom_params)

        loaded = manager.load("XAUJPY")
        assert loaded["adx_threshold"] == 10
        assert loaded["tp_mult"] == 3.0

    def test_load_merges_with_defaults(self, manager):
        """Missing params should be filled from defaults."""
        # Save only one param
        manager.save("XAUJPY", {"adx_threshold": 15})

        loaded = manager.load("XAUJPY")
        assert loaded["adx_threshold"] == 15
        # Other params should be defaults
        assert loaded["tp_mult"] == DEFAULT_PARAMS["tp_mult"]
        assert loaded["sl_mult"] == DEFAULT_PARAMS["sl_mult"]

    # ==================== Save Tests ====================

    def test_save_creates_file(self, manager, temp_config_dir):
        """Save should create a JSON file."""
        manager.save("BTCJPY", {"adx_threshold": 20})

        config_path = os.path.join(temp_config_dir, "BTCJPY.json")
        assert os.path.exists(config_path)

        with open(config_path) as f:
            data = json.load(f)
        assert data["adx_threshold"] == 20

    def test_save_validates_params(self, manager):
        """Save should clamp invalid values."""
        # adx_threshold max is 30
        manager.save("XAUJPY", {"adx_threshold": 100})

        loaded = manager.load("XAUJPY")
        assert loaded["adx_threshold"] == 30  # Clamped to max

    # ==================== Validation Tests ====================

    def test_validate_clamps_too_high(self, manager):
        """Values above max should be clamped."""
        params = {"adx_threshold": 50}  # Max is 30
        validated = manager.validate(params)
        assert validated["adx_threshold"] == 30

    def test_validate_clamps_too_low(self, manager):
        """Values below min should be clamped."""
        params = {"adx_threshold": 1}  # Min is 3
        validated = manager.validate(params)
        assert validated["adx_threshold"] == 3

    def test_validate_converts_to_int(self, manager):
        """Int params should be rounded."""
        params = {"adx_threshold": 15.7}
        validated = manager.validate(params)
        assert validated["adx_threshold"] == 16
        assert isinstance(validated["adx_threshold"], int)

    def test_validate_keeps_floats(self, manager):
        """Float params should stay as floats."""
        params = {"tp_mult": 2.5}
        validated = manager.validate(params)
        assert validated["tp_mult"] == 2.5
        assert isinstance(validated["tp_mult"], float)

    def test_is_valid_returns_true_for_valid(self, manager):
        """is_valid should return True for valid params."""
        params = {"adx_threshold": 15, "tp_mult": 2.0}
        assert manager.is_valid(params) is True

    def test_is_valid_returns_false_for_invalid(self, manager):
        """is_valid should return False for out-of-range params."""
        params = {"adx_threshold": 100}  # Max is 30
        assert manager.is_valid(params) is False

    # ==================== History Tests ====================

    def test_save_logs_history(self, manager):
        """Each save should add to history."""
        manager.save("XAUJPY", {"adx_threshold": 10}, reason="Test 1")
        manager.save("XAUJPY", {"adx_threshold": 15}, reason="Test 2")

        history = manager.get_history("XAUJPY")
        assert len(history) == 2
        assert history[0]["reason"] == "Test 1"
        assert history[1]["reason"] == "Test 2"

    def test_history_filters_by_symbol(self, manager):
        """get_history should filter by symbol."""
        manager.save("XAUJPY", {"adx_threshold": 10})
        manager.save("BTCJPY", {"adx_threshold": 20})

        xau_history = manager.get_history("XAUJPY")
        assert len(xau_history) == 1
        assert xau_history[0]["symbol"] == "XAUJPY"

    # ==================== Rollback Tests ====================

    def test_rollback_restores_previous(self, manager):
        """Rollback should restore previous parameters."""
        manager.save("XAUJPY", {"adx_threshold": 10}, reason="Version 1")
        manager.save("XAUJPY", {"adx_threshold": 20}, reason="Version 2")

        # Current should be 20
        assert manager.load("XAUJPY")["adx_threshold"] == 20

        # Rollback 1 step
        restored = manager.rollback("XAUJPY", steps=1)
        assert restored["adx_threshold"] == 10

        # Load should now return rolled back value
        assert manager.load("XAUJPY")["adx_threshold"] == 10

    def test_rollback_returns_none_if_not_enough_history(self, manager):
        """Rollback should return None if not enough history."""
        manager.save("XAUJPY", {"adx_threshold": 10})

        # Try to rollback 5 steps when only 1 exists
        result = manager.rollback("XAUJPY", steps=5)
        assert result is None


# ==================== Param Limits Tests ====================

class TestParamLimits:
    """Test that PARAM_LIMITS are sensible."""

    def test_all_default_params_are_valid(self):
        """All default params should be within limits."""
        manager = ParamManager(config_dir=tempfile.mkdtemp())
        assert manager.is_valid(DEFAULT_PARAMS) is True

    def test_limits_have_min_less_than_max(self):
        """All limits should have min < max."""
        for key, limits in PARAM_LIMITS.items():
            assert limits["min"] < limits["max"], f"{key}: min >= max"

    def test_limits_cover_all_default_params(self):
        """All default params should have limits defined."""
        for key in DEFAULT_PARAMS:
            assert key in PARAM_LIMITS, f"{key} missing from PARAM_LIMITS"


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
