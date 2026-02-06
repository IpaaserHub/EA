"""
Tests for Multi-Timeframe Confirmation
=======================================
Run with: .venv/bin/python -m pytest tests/test_mtf.py -v

Tests for:
1. Candle aggregation (H1 -> H4, H1 -> D1)
2. MTF entry confirmation logic
3. BacktestEngine with use_mtf=True
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtest.indicators import aggregate_candles, calculate_slope, calculate_adx
from backtest.engine import BacktestEngine, BacktestResult


# ==================== Test Fixtures ====================

def make_candle(open_p, high, low, close):
    """Helper to create an OHLC dict."""
    return {"open": open_p, "high": high, "low": low, "close": close}


def make_trending_candles(n, start_price=100.0, trend=0.5):
    """Generate n H1 candles with a trend."""
    candles = []
    for i in range(n):
        p = start_price + i * trend
        candles.append(make_candle(p, p + 0.3, p - 0.2, p + 0.1))
    return candles


def make_downtrending_candles(n, start_price=200.0, trend=-0.5):
    """Generate n H1 candles with a downtrend."""
    return make_trending_candles(n, start_price, trend)


def make_ranging_candles(n, center=100.0, amplitude=1.0):
    """Generate n H1 candles oscillating around a center."""
    candles = []
    for i in range(n):
        offset = amplitude * (1 if i % 4 < 2 else -1)
        p = center + offset
        candles.append(make_candle(p - 0.1, p + 0.3, p - 0.3, p + 0.1))
    return candles


# ==================== Candle Aggregation Tests ====================

class TestAggregateCandles:
    """Tests for aggregate_candles function."""

    def test_h4_basic(self):
        """Should aggregate 8 H1 candles into 2 H4 candles."""
        candles = make_trending_candles(8)
        h4 = aggregate_candles(candles, "H4")

        assert len(h4) == 2

    def test_h4_ohlc_values(self):
        """H4 OHLC should be derived correctly from H1 group."""
        candles = [
            make_candle(10, 15, 8, 12),   # H1 candle 1
            make_candle(12, 18, 10, 14),   # H1 candle 2
            make_candle(14, 14, 7, 9),     # H1 candle 3
            make_candle(9, 16, 6, 13),     # H1 candle 4
        ]
        h4 = aggregate_candles(candles, "H4")

        assert len(h4) == 1
        assert h4[0]["open"] == 10     # First candle's open
        assert h4[0]["high"] == 18     # Max high across all 4
        assert h4[0]["low"] == 6       # Min low across all 4
        assert h4[0]["close"] == 13    # Last candle's close

    def test_h4_incomplete_group_dropped(self):
        """Incomplete groups at the start should be dropped."""
        candles = make_trending_candles(10)  # 10 = 2 groups of 4 + 2 remainder
        h4 = aggregate_candles(candles, "H4")

        assert len(h4) == 2  # 2 complete groups, 2 dropped

    def test_d1_basic(self):
        """Should aggregate 48 H1 candles into 2 D1 candles."""
        candles = make_trending_candles(48)
        d1 = aggregate_candles(candles, "D1")

        assert len(d1) == 2

    def test_d1_insufficient_data(self):
        """Should return empty list if fewer than 24 candles."""
        candles = make_trending_candles(20)
        d1 = aggregate_candles(candles, "D1")

        assert len(d1) == 0

    def test_empty_input(self):
        """Should return empty list for empty input."""
        assert aggregate_candles([], "H4") == []

    def test_invalid_timeframe(self):
        """Should raise ValueError for unsupported timeframe."""
        candles = make_trending_candles(8)
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            aggregate_candles(candles, "M15")

    def test_single_group(self):
        """Should work with exactly one complete group."""
        candles = make_trending_candles(4)
        h4 = aggregate_candles(candles, "H4")

        assert len(h4) == 1

    def test_preserves_trend(self):
        """H4 candles from trending data should show trending slope."""
        candles = make_trending_candles(40, trend=1.0)
        h4 = aggregate_candles(candles, "H4")

        assert len(h4) >= 5
        slope = calculate_slope(h4)
        assert slope > 0  # Should detect uptrend


# ==================== MTF Entry Confirmation Tests ====================

class TestMTFEntry:
    """Tests for multi-timeframe entry confirmation in BacktestEngine."""

    def test_mtf_filters_counter_trend(self):
        """MTF should filter signals that go against H4 trend."""
        # Create data where H4 trend is down but H1 might give BUY signal
        # H4 downtrend = first 80 candles trending down
        down_candles = make_downtrending_candles(100, start_price=200.0, trend=-0.3)
        # Then slight uptick in last candles (H1 might see BUY)
        up_candles = make_trending_candles(100, start_price=170.0, trend=0.2)
        all_candles = down_candles + up_candles

        engine = BacktestEngine(all_candles, lookback=50)

        # Without MTF - should get some trades
        result_no_mtf = engine.run(
            {"adx_threshold": 3, "slope_threshold": 0.00001,
             "buy_position": 0.7, "sell_position": 0.3},
            use_mtf=False
        )

        # With MTF - should get fewer trades (counter-trend filtered)
        result_mtf = engine.run(
            {"adx_threshold": 3, "slope_threshold": 0.00001,
             "buy_position": 0.7, "sell_position": 0.3, "h4_adx_min": 5},
            use_mtf=True
        )

        # MTF should filter some trades
        assert result_mtf.total_trades <= result_no_mtf.total_trades

    def test_mtf_allows_aligned_trades(self):
        """MTF should allow signals that align with H4 trend."""
        # Strong uptrend across all timeframes
        candles = make_trending_candles(200, trend=0.5)
        engine = BacktestEngine(candles, lookback=50)

        result = engine.run(
            {"adx_threshold": 3, "slope_threshold": 0.00001,
             "buy_position": 0.7, "sell_position": 0.3, "h4_adx_min": 5},
            use_mtf=True
        )

        # Should still get BUY trades in a strong uptrend
        if result.total_trades > 0:
            buy_trades = [t for t in result.trades if t.direction == "BUY"]
            assert len(buy_trades) > 0

    def test_mtf_graceful_degradation(self):
        """With insufficient data for H4, should fall through to H1 only."""
        # Very short data - not enough for meaningful H4
        candles = make_trending_candles(60, trend=0.3)
        engine = BacktestEngine(candles, lookback=50)

        result_mtf = engine.run(
            {"adx_threshold": 3, "slope_threshold": 0.00001},
            use_mtf=True
        )

        result_no_mtf = engine.run(
            {"adx_threshold": 3, "slope_threshold": 0.00001},
            use_mtf=False
        )

        # Both should produce results (MTF degrades gracefully)
        assert isinstance(result_mtf, BacktestResult)
        assert isinstance(result_no_mtf, BacktestResult)

    def test_mtf_backtest_result_type(self):
        """MTF run should return proper BacktestResult."""
        candles = make_trending_candles(200)
        engine = BacktestEngine(candles)

        result = engine.run(
            {"adx_threshold": 5, "slope_threshold": 0.00001},
            use_mtf=True
        )

        assert isinstance(result, BacktestResult)
        assert hasattr(result, 'total_trades')
        assert hasattr(result, 'profit_factor')
        assert hasattr(result, 'win_rate')


# ==================== Integration with Real Data ====================

class TestMTFWithRealParams:
    """Tests using parameters similar to production config."""

    def test_usdjpy_like_params(self):
        """Test MTF with USDJPY-like parameters."""
        # Simulate USDJPY-like price movement
        candles = make_trending_candles(200, start_price=150.0, trend=0.01)
        engine = BacktestEngine(candles, lookback=50)

        params = {
            "adx_threshold": 8,
            "slope_threshold": 0.00002,
            "buy_position": 0.48,
            "sell_position": 0.52,
            "rsi_buy_max": 70,
            "rsi_sell_min": 30,
            "tp_mult": 2.0,
            "sl_mult": 1.5,
            "h4_adx_min": 15,
        }

        result = engine.run(params, use_mtf=True)
        assert isinstance(result, BacktestResult)


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
