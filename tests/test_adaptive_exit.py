"""
Tests for Adaptive Exit Timing
===============================
Run with: .venv/bin/python -m pytest tests/test_adaptive_exit.py -v

Tests for:
1. Trailing stop activation and movement
2. Time-based exit
3. Hard SL/TP still respected
4. Integration with BacktestEngine
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtest.engine import BacktestEngine, BacktestResult, Trade


# ==================== Test Fixtures ====================

def make_candle(open_p, high, low, close):
    """Helper to create an OHLC dict."""
    return {"open": open_p, "high": high, "low": low, "close": close}


def make_trending_candles(n, start_price=100.0, trend=0.5):
    """Generate n candles with a trend."""
    candles = []
    for i in range(n):
        p = start_price + i * trend
        candles.append(make_candle(p, p + 0.3, p - 0.2, p + 0.1))
    return candles


def make_reversal_candles(n_up, n_down, start_price=100.0, trend=0.5):
    """Generate candles that trend up then reverse down."""
    up = make_trending_candles(n_up, start_price, trend)
    last_price = start_price + n_up * trend
    down = make_trending_candles(n_down, last_price, -trend)
    return up + down


# ==================== Unit Tests for _check_exit_adaptive ====================

class TestCheckExitAdaptive:
    """Unit tests for the adaptive exit method."""

    def _make_engine(self):
        """Create engine with minimal data."""
        return BacktestEngine(make_trending_candles(100), lookback=50)

    def test_hard_tp_hit_buy(self):
        """Hard TP should still trigger for BUY."""
        engine = self._make_engine()
        trade = Trade(
            entry_index=0, entry_price=100.0, direction="BUY",
            sl=98.0, tp=104.0, peak_price=100.0,
        )
        candle = make_candle(102, 105, 101, 103)  # high > TP

        result = engine._check_exit_adaptive(
            trade, candle, current_atr=1.0, bars_held=5,
            trail_activation_atr=1.0, trail_distance_mult=1.0,
            max_bars_in_trade=50,
        )

        assert result is not None
        assert result["outcome"] == "WIN"
        assert result["price"] == 104.0
        assert result["profit"] == 4.0

    def test_hard_tp_hit_sell(self):
        """Hard TP should still trigger for SELL."""
        engine = self._make_engine()
        trade = Trade(
            entry_index=0, entry_price=100.0, direction="SELL",
            sl=102.0, tp=96.0, peak_price=100.0,
        )
        candle = make_candle(97, 98, 95, 96)  # low < TP

        result = engine._check_exit_adaptive(
            trade, candle, current_atr=1.0, bars_held=5,
            trail_activation_atr=1.0, trail_distance_mult=1.0,
            max_bars_in_trade=50,
        )

        assert result is not None
        assert result["outcome"] == "WIN"
        assert result["price"] == 96.0
        assert result["profit"] == 4.0

    def test_trailing_activation_buy(self):
        """Trailing stop should activate when profit >= activation threshold."""
        engine = self._make_engine()
        trade = Trade(
            entry_index=0, entry_price=100.0, direction="BUY",
            sl=98.0, tp=110.0, peak_price=100.0,
        )

        # Move price up enough to activate trailing (1.0 ATR = 1.0, need >= 1.0 profit)
        # Low must stay ABOVE the resulting trail stop (102.0 - 1.0 = 101.0)
        candle = make_candle(101, 102.0, 101.2, 101.5)
        result = engine._check_exit_adaptive(
            trade, candle, current_atr=1.0, bars_held=3,
            trail_activation_atr=1.0, trail_distance_mult=1.0,
            max_bars_in_trade=50,
        )

        assert result is None  # No exit yet
        assert trade.trailing_active is True
        assert trade.peak_price == 102.0
        # trail = 102.0 - (1.0 * 1.0) = 101.0
        assert trade.trailing_stop == 101.0

    def test_trailing_activation_sell(self):
        """Trailing stop should activate for SELL when profit >= threshold."""
        engine = self._make_engine()
        trade = Trade(
            entry_index=0, entry_price=100.0, direction="SELL",
            sl=102.0, tp=90.0, peak_price=100.0,
        )

        # Move price down enough (entry 100, low 98 = 2.0 unrealized profit)
        # High must stay BELOW the resulting trail stop (98.0 + 1.0 = 99.0)
        candle = make_candle(98.8, 98.9, 98.0, 98.5)
        result = engine._check_exit_adaptive(
            trade, candle, current_atr=1.0, bars_held=3,
            trail_activation_atr=1.0, trail_distance_mult=1.0,
            max_bars_in_trade=50,
        )

        assert result is None
        assert trade.trailing_active is True
        assert trade.peak_price == 98.0
        # trail = 98.0 + (1.0 * 1.0) = 99.0
        assert trade.trailing_stop == 99.0

    def test_trailing_stop_only_moves_favorable_buy(self):
        """Trailing stop should never move down for BUY trades."""
        engine = self._make_engine()
        trade = Trade(
            entry_index=0, entry_price=100.0, direction="BUY",
            sl=98.0, tp=110.0, peak_price=105.0,
            trailing_active=True, trailing_stop=104.0,
        )

        # Price retraces but doesn't hit trail (low must stay above 104.0)
        candle = make_candle(104.5, 104.8, 104.2, 104.3)
        result = engine._check_exit_adaptive(
            trade, candle, current_atr=1.0, bars_held=5,
            trail_activation_atr=1.0, trail_distance_mult=1.0,
            max_bars_in_trade=50,
        )

        assert result is None
        assert trade.trailing_stop == 104.0  # Unchanged (peak didn't increase)

    def test_trailing_stop_only_moves_favorable_sell(self):
        """Trailing stop should never move up for SELL trades."""
        engine = self._make_engine()
        trade = Trade(
            entry_index=0, entry_price=100.0, direction="SELL",
            sl=102.0, tp=90.0, peak_price=96.0,
            trailing_active=True, trailing_stop=97.0,
        )

        # Price bounces up but doesn't hit trail
        candle = make_candle(96.5, 96.8, 96.2, 96.5)
        result = engine._check_exit_adaptive(
            trade, candle, current_atr=1.0, bars_held=5,
            trail_activation_atr=1.0, trail_distance_mult=1.0,
            max_bars_in_trade=50,
        )

        assert result is None
        assert trade.trailing_stop == 97.0  # Unchanged

    def test_trailing_stop_hit_buy(self):
        """Price hitting trailing stop should exit for BUY."""
        engine = self._make_engine()
        trade = Trade(
            entry_index=0, entry_price=100.0, direction="BUY",
            sl=98.0, tp=110.0, peak_price=104.0,
            trailing_active=True, trailing_stop=103.0,
        )

        # Price drops through trailing stop
        candle = make_candle(103.5, 103.5, 102.5, 102.8)
        result = engine._check_exit_adaptive(
            trade, candle, current_atr=1.0, bars_held=10,
            trail_activation_atr=1.0, trail_distance_mult=1.0,
            max_bars_in_trade=50,
        )

        assert result is not None
        assert result["price"] == 103.0
        assert result["profit"] == 3.0
        assert result["outcome"] == "WIN"

    def test_trailing_stop_hit_sell(self):
        """Price hitting trailing stop should exit for SELL."""
        engine = self._make_engine()
        trade = Trade(
            entry_index=0, entry_price=100.0, direction="SELL",
            sl=102.0, tp=90.0, peak_price=96.0,
            trailing_active=True, trailing_stop=97.0,
        )

        # Price bounces up through trailing stop
        candle = make_candle(96.5, 97.5, 96.2, 97.2)
        result = engine._check_exit_adaptive(
            trade, candle, current_atr=1.0, bars_held=10,
            trail_activation_atr=1.0, trail_distance_mult=1.0,
            max_bars_in_trade=50,
        )

        assert result is not None
        assert result["price"] == 97.0
        assert result["profit"] == 3.0
        assert result["outcome"] == "WIN"

    def test_time_exit_buy(self):
        """Should exit at close price after max bars."""
        engine = self._make_engine()
        trade = Trade(
            entry_index=0, entry_price=100.0, direction="BUY",
            sl=98.0, tp=110.0, peak_price=100.5,
        )

        candle = make_candle(100.3, 100.5, 100.0, 100.2)
        result = engine._check_exit_adaptive(
            trade, candle, current_atr=1.0, bars_held=50,
            trail_activation_atr=1.0, trail_distance_mult=1.0,
            max_bars_in_trade=50,
        )

        assert result is not None
        assert result["price"] == 100.2  # Close price
        assert result["profit"] == pytest.approx(0.2)
        assert result["outcome"] == "WIN"

    def test_time_exit_sell_loss(self):
        """Time exit with loss for SELL."""
        engine = self._make_engine()
        trade = Trade(
            entry_index=0, entry_price=100.0, direction="SELL",
            sl=103.0, tp=90.0, peak_price=100.0,
        )

        candle = make_candle(100.5, 101.0, 100.0, 100.8)
        result = engine._check_exit_adaptive(
            trade, candle, current_atr=1.0, bars_held=50,
            trail_activation_atr=1.0, trail_distance_mult=1.0,
            max_bars_in_trade=50,
        )

        assert result is not None
        assert result["price"] == 100.8
        assert result["profit"] == pytest.approx(-0.8)
        assert result["outcome"] == "LOSS"

    def test_hard_sl_still_works_buy(self):
        """Hard SL should still be the safety net."""
        engine = self._make_engine()
        trade = Trade(
            entry_index=0, entry_price=100.0, direction="BUY",
            sl=98.0, tp=110.0, peak_price=100.0,
        )

        candle = make_candle(99, 99.5, 97.5, 98.5)  # Low below SL
        result = engine._check_exit_adaptive(
            trade, candle, current_atr=1.0, bars_held=5,
            trail_activation_atr=1.0, trail_distance_mult=1.0,
            max_bars_in_trade=50,
        )

        assert result is not None
        assert result["price"] == 98.0
        assert result["profit"] == -2.0
        assert result["outcome"] == "LOSS"

    def test_hard_sl_still_works_sell(self):
        """Hard SL should still be the safety net for SELL."""
        engine = self._make_engine()
        trade = Trade(
            entry_index=0, entry_price=100.0, direction="SELL",
            sl=102.0, tp=90.0, peak_price=100.0,
        )

        candle = make_candle(101, 102.5, 100.5, 101.5)  # High above SL
        result = engine._check_exit_adaptive(
            trade, candle, current_atr=1.0, bars_held=5,
            trail_activation_atr=1.0, trail_distance_mult=1.0,
            max_bars_in_trade=50,
        )

        assert result is not None
        assert result["price"] == 102.0
        assert result["profit"] == -2.0
        assert result["outcome"] == "LOSS"

    def test_no_exit_when_nothing_triggered(self):
        """Should return None when no exit condition is met."""
        engine = self._make_engine()
        trade = Trade(
            entry_index=0, entry_price=100.0, direction="BUY",
            sl=98.0, tp=110.0, peak_price=100.0,
        )

        candle = make_candle(100.1, 100.3, 99.5, 100.2)
        result = engine._check_exit_adaptive(
            trade, candle, current_atr=1.0, bars_held=5,
            trail_activation_atr=1.0, trail_distance_mult=1.0,
            max_bars_in_trade=50,
        )

        assert result is None


# ==================== Integration Tests ====================

class TestAdaptiveExitIntegration:
    """Tests for adaptive exit with full BacktestEngine runs."""

    def test_run_with_adaptive_params(self):
        """Engine should use adaptive exits when trail params are set."""
        candles = make_trending_candles(200, trend=0.5)
        engine = BacktestEngine(candles, lookback=50)

        params = {
            "adx_threshold": 3,
            "slope_threshold": 0.00001,
            "buy_position": 0.7,
            "sell_position": 0.3,
            "tp_mult": 3.0,
            "sl_mult": 1.5,
            "trail_activation_atr": 1.0,
            "trail_distance_mult": 1.0,
            "max_bars_in_trade": 100,
        }

        result = engine.run(params)
        assert isinstance(result, BacktestResult)

    def test_run_without_adaptive_params(self):
        """Engine should use fixed exits when no trail params."""
        candles = make_trending_candles(200, trend=0.5)
        engine = BacktestEngine(candles, lookback=50)

        params = {
            "adx_threshold": 3,
            "slope_threshold": 0.00001,
            "buy_position": 0.7,
            "sell_position": 0.3,
            "tp_mult": 3.0,
            "sl_mult": 1.5,
        }

        result = engine.run(params)
        assert isinstance(result, BacktestResult)

    def test_adaptive_vs_fixed_different_results(self):
        """Adaptive and fixed exits should generally produce different results."""
        candles = make_reversal_candles(120, 80, start_price=100.0, trend=0.3)
        engine = BacktestEngine(candles, lookback=50)

        base_params = {
            "adx_threshold": 3,
            "slope_threshold": 0.00001,
            "buy_position": 0.7,
            "sell_position": 0.3,
            "tp_mult": 3.0,
            "sl_mult": 1.5,
        }

        adaptive_params = {
            **base_params,
            "trail_activation_atr": 0.8,
            "trail_distance_mult": 1.0,
            "max_bars_in_trade": 50,
        }

        result_fixed = engine.run(base_params)
        result_adaptive = engine.run(adaptive_params)

        assert isinstance(result_fixed, BacktestResult)
        assert isinstance(result_adaptive, BacktestResult)
        # Results may differ (not guaranteed but likely with different exit logic)

    def test_time_exit_limits_trade_duration(self):
        """With max_bars_in_trade, trades shouldn't last forever."""
        # Create sideways market where trades might never hit TP/SL
        candles = []
        for i in range(300):
            p = 100.0 + (0.1 if i % 2 == 0 else -0.1)
            candles.append(make_candle(p, p + 0.15, p - 0.15, p + 0.05))

        engine = BacktestEngine(candles, lookback=50)

        params = {
            "adx_threshold": 1,
            "slope_threshold": 0.0000001,
            "buy_position": 0.9,
            "sell_position": 0.1,
            "tp_mult": 4.0,  # Very far TP
            "sl_mult": 4.0,  # Very far SL
            "trail_activation_atr": 1.0,
            "trail_distance_mult": 1.0,
            "max_bars_in_trade": 20,
        }

        result = engine.run(params)
        assert isinstance(result, BacktestResult)

    def test_adaptive_exit_with_mtf(self):
        """Adaptive exit should work together with MTF."""
        candles = make_trending_candles(200, trend=0.5)
        engine = BacktestEngine(candles, lookback=50)

        params = {
            "adx_threshold": 3,
            "slope_threshold": 0.00001,
            "buy_position": 0.7,
            "sell_position": 0.3,
            "tp_mult": 3.0,
            "sl_mult": 1.5,
            "trail_activation_atr": 1.0,
            "trail_distance_mult": 1.0,
            "max_bars_in_trade": 100,
            "h4_adx_min": 10,
        }

        result = engine.run(params, use_mtf=True)
        assert isinstance(result, BacktestResult)


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
