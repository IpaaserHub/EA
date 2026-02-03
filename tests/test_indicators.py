"""
Tests for Technical Indicators
==============================
Run with: pytest tests/test_indicators.py -v

These tests verify that indicators:
1. Return sensible values
2. Handle edge cases (not enough data)
3. Match expected behavior
"""

import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtest.indicators import (
    calculate_rsi,
    calculate_adx,
    calculate_atr,
    calculate_slope,
    calculate_position_in_range,
    calculate_all_indicators,
)


# ==================== Test Data Fixtures ====================

@pytest.fixture
def uptrend_prices():
    """Generate prices in an uptrend."""
    prices = []
    base = 100.0
    for i in range(100):
        price = base + i * 0.5  # Steadily increasing
        prices.append({
            "open": price - 0.1,
            "high": price + 0.2,
            "low": price - 0.2,
            "close": price,
        })
    return prices


@pytest.fixture
def downtrend_prices():
    """Generate prices in a downtrend."""
    prices = []
    base = 150.0
    for i in range(100):
        price = base - i * 0.5  # Steadily decreasing
        prices.append({
            "open": price + 0.1,
            "high": price + 0.2,
            "low": price - 0.2,
            "close": price,
        })
    return prices


@pytest.fixture
def sideways_prices():
    """Generate prices moving sideways (range)."""
    prices = []
    import math
    for i in range(100):
        # Oscillate around 100
        price = 100 + 2 * math.sin(i * 0.3)
        prices.append({
            "open": price - 0.1,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
        })
    return prices


@pytest.fixture
def minimal_prices():
    """Just a few prices - not enough for most indicators."""
    return [
        {"open": 100, "high": 101, "low": 99, "close": 100.5},
        {"open": 100.5, "high": 102, "low": 100, "close": 101},
    ]


# ==================== RSI Tests ====================

class TestRSI:
    """Test RSI indicator."""

    def test_rsi_uptrend_is_high(self, uptrend_prices):
        """RSI should be high (>50) in an uptrend."""
        rsi = calculate_rsi(uptrend_prices)
        assert rsi > 50, f"RSI in uptrend should be >50, got {rsi}"

    def test_rsi_downtrend_is_low(self, downtrend_prices):
        """RSI should be low (<50) in a downtrend."""
        rsi = calculate_rsi(downtrend_prices)
        assert rsi < 50, f"RSI in downtrend should be <50, got {rsi}"

    def test_rsi_range_is_valid(self, uptrend_prices):
        """RSI should always be between 0 and 100."""
        rsi = calculate_rsi(uptrend_prices)
        assert 0 <= rsi <= 100

    def test_rsi_returns_neutral_with_minimal_data(self, minimal_prices):
        """RSI should return 50 when not enough data."""
        rsi = calculate_rsi(minimal_prices)
        assert rsi == 50.0


# ==================== ADX Tests ====================

class TestADX:
    """Test ADX indicator."""

    def test_adx_trending_is_high(self, uptrend_prices):
        """ADX should be higher in a trending market."""
        adx = calculate_adx(uptrend_prices)
        assert adx > 15, f"ADX in trend should be >15, got {adx}"

    def test_adx_range_is_valid(self, uptrend_prices):
        """ADX should always be between 0 and 100."""
        adx = calculate_adx(uptrend_prices)
        assert 0 <= adx <= 100

    def test_adx_returns_neutral_with_minimal_data(self, minimal_prices):
        """ADX should return 20 when not enough data."""
        adx = calculate_adx(minimal_prices)
        assert adx == 20.0


# ==================== ATR Tests ====================

class TestATR:
    """Test ATR indicator."""

    def test_atr_is_positive(self, uptrend_prices):
        """ATR should always be positive."""
        atr = calculate_atr(uptrend_prices)
        assert atr > 0

    def test_atr_returns_small_with_minimal_data(self, minimal_prices):
        """ATR should return small default when not enough data."""
        atr = calculate_atr(minimal_prices)
        assert atr == 0.01


# ==================== Slope Tests ====================

class TestSlope:
    """Test slope indicator."""

    def test_slope_uptrend_is_positive(self, uptrend_prices):
        """Slope should be positive in an uptrend."""
        slope = calculate_slope(uptrend_prices)
        assert slope > 0, f"Slope in uptrend should be >0, got {slope}"

    def test_slope_downtrend_is_negative(self, downtrend_prices):
        """Slope should be negative in a downtrend."""
        slope = calculate_slope(downtrend_prices)
        assert slope < 0, f"Slope in downtrend should be <0, got {slope}"

    def test_slope_sideways_is_near_zero(self, sideways_prices):
        """Slope should be near zero in sideways market."""
        slope = calculate_slope(sideways_prices)
        assert abs(slope) < 0.1, f"Slope in sideways should be near 0, got {slope}"


# ==================== Position in Range Tests ====================

class TestPositionInRange:
    """Test position in range indicator."""

    def test_position_uptrend_at_top(self, uptrend_prices):
        """In uptrend, current price should be near top of recent range."""
        position = calculate_position_in_range(uptrend_prices)
        assert position > 0.5, f"Position in uptrend should be >0.5, got {position}"

    def test_position_downtrend_at_bottom(self, downtrend_prices):
        """In downtrend, current price should be near bottom of recent range."""
        position = calculate_position_in_range(downtrend_prices)
        assert position < 0.5, f"Position in downtrend should be <0.5, got {position}"

    def test_position_range_is_valid(self, uptrend_prices):
        """Position should always be between 0 and 1."""
        position = calculate_position_in_range(uptrend_prices)
        assert 0 <= position <= 1


# ==================== All Indicators Tests ====================

class TestAllIndicators:
    """Test calculate_all_indicators function."""

    def test_returns_all_keys(self, uptrend_prices):
        """Should return all expected indicator keys."""
        result = calculate_all_indicators(uptrend_prices)

        expected_keys = ["rsi", "adx", "atr", "slope", "position"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_values_are_numbers(self, uptrend_prices):
        """All values should be numbers."""
        result = calculate_all_indicators(uptrend_prices)

        for key, value in result.items():
            assert isinstance(value, (int, float)), f"{key} is not a number"


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
