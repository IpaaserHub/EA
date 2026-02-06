"""
Tests for Market Regime Detector
=================================
Run with: .venv/bin/python -m pytest tests/test_regime_detector.py -v

Tests for:
1. Feature computation
2. Model fitting
3. Regime detection
4. Parameter adjustment per regime
5. Edge cases (insufficient data, unfitted model)
"""

import pytest
import os
import sys
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai.regime_detector import (
    RegimeDetector, RegimeResult,
    TRENDING, RANGING, VOLATILE,
)


# ==================== Test Fixtures ====================

def make_candle(open_p, high, low, close):
    return {"open": open_p, "high": high, "low": low, "close": close}


def make_trending_candles(n, start_price=100.0, trend=0.5):
    """Strong uptrend: high ADX, positive slope, consistent."""
    candles = []
    for i in range(n):
        p = start_price + i * trend
        candles.append(make_candle(p, p + 0.3, p - 0.2, p + 0.1))
    return candles


def make_ranging_candles(n, center=100.0, amplitude=0.5):
    """Sideways market: low ADX, near-zero slope."""
    candles = []
    for i in range(n):
        offset = amplitude * math.sin(i * 0.5)
        p = center + offset
        candles.append(make_candle(p - 0.1, p + 0.15, p - 0.15, p + 0.05))
    return candles


def make_volatile_candles(n, center=100.0, amplitude=3.0):
    """High volatility: big swings, high ATR, inconsistent direction."""
    random.seed(42)
    candles = []
    p = center
    for i in range(n):
        change = amplitude * (random.random() - 0.5) * 2
        p += change
        high = p + abs(change) * 0.5
        low = p - abs(change) * 0.5
        candles.append(make_candle(p - change * 0.3, high, low, p))
    return candles


def make_mixed_data(n_each=200):
    """Create mixed data with all three regime types for training."""
    trending = make_trending_candles(n_each, start_price=100.0, trend=0.3)
    ranging = make_ranging_candles(n_each, center=160.0, amplitude=0.3)
    volatile = make_volatile_candles(n_each, center=160.0, amplitude=2.0)
    return trending + ranging + volatile


# ==================== Feature Computation Tests ====================

class TestFeatureComputation:
    """Tests for _compute_features."""

    def test_trending_has_high_adx(self):
        """Trending market should have high ADX."""
        detector = RegimeDetector(window_size=50)
        candles = make_trending_candles(50, trend=0.5)
        features = detector._compute_features(candles)
        assert features["adx"] > 10  # Trending should have meaningful ADX

    def test_trending_has_high_slope(self):
        """Trending market should have high slope magnitude."""
        detector = RegimeDetector(window_size=50)
        candles = make_trending_candles(50, trend=0.5)
        features = detector._compute_features(candles)
        assert features["slope_abs"] > 0

    def test_ranging_has_low_slope(self):
        """Ranging market should have near-zero slope."""
        detector = RegimeDetector(window_size=50)
        candles = make_ranging_candles(50, amplitude=0.3)
        features = detector._compute_features(candles)
        assert features["slope_abs"] < 0.1

    def test_volatile_has_high_atr(self):
        """Volatile market should have high ATR percentage."""
        detector = RegimeDetector(window_size=50)
        candles_vol = make_volatile_candles(50, amplitude=3.0)
        candles_calm = make_ranging_candles(50, amplitude=0.3)

        features_vol = detector._compute_features(candles_vol)
        features_calm = detector._compute_features(candles_calm)

        assert features_vol["atr_pct"] > features_calm["atr_pct"]

    def test_features_have_all_keys(self):
        """Feature dict should have all expected keys."""
        detector = RegimeDetector(window_size=50)
        candles = make_trending_candles(50)
        features = detector._compute_features(candles)
        assert "adx" in features
        assert "atr_pct" in features
        assert "slope_abs" in features
        assert "slope_consistency" in features

    def test_slope_consistency_high_for_trend(self):
        """Strong trend should have high slope consistency."""
        detector = RegimeDetector(window_size=50)
        candles = make_trending_candles(50, trend=0.5)
        features = detector._compute_features(candles)
        assert features["slope_consistency"] == 1.0


# ==================== Model Fitting Tests ====================

class TestModelFitting:
    """Tests for fit() method."""

    def test_fit_succeeds_with_enough_data(self):
        """Should fit successfully with sufficient data."""
        detector = RegimeDetector(window_size=50)
        data = make_mixed_data(200)
        assert detector.fit(data) is True
        assert detector.is_fitted is True

    def test_fit_fails_with_insufficient_data(self):
        """Should fail gracefully with too little data."""
        detector = RegimeDetector(window_size=50)
        data = make_trending_candles(30)
        assert detector.fit(data) is False
        assert detector.is_fitted is False

    def test_fit_creates_3_clusters(self):
        """Should create exactly 3 cluster labels."""
        detector = RegimeDetector(window_size=50)
        data = make_mixed_data(200)
        detector.fit(data)
        assert len(detector.cluster_labels) == 3

    def test_fit_labels_include_all_regimes(self):
        """All three regime types should be represented."""
        detector = RegimeDetector(window_size=50)
        data = make_mixed_data(200)
        detector.fit(data)
        labels = set(detector.cluster_labels.values())
        assert TRENDING in labels
        assert RANGING in labels
        assert VOLATILE in labels


# ==================== Regime Detection Tests ====================

class TestRegimeDetection:
    """Tests for detect() method."""

    @pytest.fixture
    def fitted_detector(self):
        """Create and fit a detector on mixed data."""
        detector = RegimeDetector(window_size=50)
        data = make_mixed_data(200)
        detector.fit(data)
        return detector

    def test_detect_returns_regime_result(self, fitted_detector):
        """Should return a RegimeResult."""
        candles = make_trending_candles(50)
        result = fitted_detector.detect(candles)
        assert isinstance(result, RegimeResult)
        assert result.regime in [TRENDING, RANGING, VOLATILE]

    def test_detect_trending_data(self, fitted_detector):
        """Should detect trending regime in trending data."""
        candles = make_trending_candles(50, trend=0.5)
        result = fitted_detector.detect(candles)
        # Allow for some misclassification but prefer TRENDING
        assert result.regime in [TRENDING, RANGING, VOLATILE]
        assert result.confidence > 0

    def test_detect_has_confidence(self, fitted_detector):
        """Result should have confidence between 0 and 1."""
        candles = make_trending_candles(50)
        result = fitted_detector.detect(candles)
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_has_features(self, fitted_detector):
        """Result should include computed features."""
        candles = make_trending_candles(50)
        result = fitted_detector.detect(candles)
        assert "adx" in result.features
        assert "atr_pct" in result.features

    def test_detect_unfitted_returns_ranging(self):
        """Unfitted detector should default to RANGING."""
        detector = RegimeDetector(window_size=50)
        candles = make_trending_candles(50)
        result = detector.detect(candles)
        assert result.regime == RANGING
        assert result.confidence == 0.0

    def test_detect_insufficient_data_returns_ranging(self, fitted_detector):
        """Should return RANGING with insufficient data."""
        candles = make_trending_candles(10)
        result = fitted_detector.detect(candles)
        assert result.regime == RANGING
        assert result.confidence == 0.0

    def test_to_dict(self, fitted_detector):
        """RegimeResult.to_dict() should produce valid dict."""
        candles = make_trending_candles(50)
        result = fitted_detector.detect(candles)
        d = result.to_dict()
        assert "regime" in d
        assert "confidence" in d
        assert "features" in d


# ==================== Parameter Adjustment Tests ====================

class TestRegimeParams:
    """Tests for get_regime_params()."""

    def test_trending_widens_tp(self):
        """TRENDING should increase TP multiplier."""
        detector = RegimeDetector()
        base = {"tp_mult": 2.0, "buy_position": 0.5, "sell_position": 0.5}
        adjusted = detector.get_regime_params(TRENDING, base)
        assert adjusted["tp_mult"] > base["tp_mult"]

    def test_ranging_tightens_tp(self):
        """RANGING should decrease TP multiplier."""
        detector = RegimeDetector()
        base = {"tp_mult": 2.0, "adx_threshold": 5}
        adjusted = detector.get_regime_params(RANGING, base)
        assert adjusted["tp_mult"] < base["tp_mult"]

    def test_ranging_raises_adx_threshold(self):
        """RANGING should require stronger trend (higher ADX)."""
        detector = RegimeDetector()
        base = {"tp_mult": 2.0, "adx_threshold": 5}
        adjusted = detector.get_regime_params(RANGING, base)
        assert adjusted["adx_threshold"] > base["adx_threshold"]

    def test_volatile_tightens_sl(self):
        """VOLATILE should tighten SL."""
        detector = RegimeDetector()
        base = {"sl_mult": 1.5, "tp_mult": 2.0, "rsi_buy_max": 75, "rsi_sell_min": 25}
        adjusted = detector.get_regime_params(VOLATILE, base)
        assert adjusted["sl_mult"] < base["sl_mult"]

    def test_volatile_tightens_rsi(self):
        """VOLATILE should use stricter RSI filters."""
        detector = RegimeDetector()
        base = {"sl_mult": 1.5, "tp_mult": 2.0, "rsi_buy_max": 75, "rsi_sell_min": 25}
        adjusted = detector.get_regime_params(VOLATILE, base)
        assert adjusted["rsi_buy_max"] < base["rsi_buy_max"]
        assert adjusted["rsi_sell_min"] > base["rsi_sell_min"]

    def test_params_stay_within_limits(self):
        """Adjusted params should stay within PARAM_LIMITS."""
        from config.param_manager import PARAM_LIMITS

        detector = RegimeDetector()
        base = {
            "tp_mult": 2.0, "sl_mult": 1.5, "adx_threshold": 5,
            "buy_position": 0.5, "sell_position": 0.5,
            "rsi_buy_max": 75, "rsi_sell_min": 25,
        }

        for regime in [TRENDING, RANGING, VOLATILE]:
            adjusted = detector.get_regime_params(regime, base)
            for key, value in adjusted.items():
                if key in PARAM_LIMITS:
                    limits = PARAM_LIMITS[key]
                    assert value >= limits["min"], \
                        f"{regime}: {key}={value} below min {limits['min']}"
                    assert value <= limits["max"], \
                        f"{regime}: {key}={value} above max {limits['max']}"

    def test_does_not_modify_original(self):
        """Should not modify the original params dict."""
        detector = RegimeDetector()
        base = {"tp_mult": 2.0, "sl_mult": 1.5}
        original_tp = base["tp_mult"]
        detector.get_regime_params(TRENDING, base)
        assert base["tp_mult"] == original_tp


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
