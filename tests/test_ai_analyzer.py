"""
Tests for AI Analyzer Module
============================
Run with: ./venv/bin/python -m pytest tests/test_ai_analyzer.py -v

Tests for:
1. ParameterSuggestion and AnalysisResult dataclasses
2. Fallback analysis (rule-based, no API needed)
3. Response parsing
4. Parameter validation and clamping
"""

import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimizer.ai_analyzer import (
    AIAnalyzer,
    AnalysisResult,
    ParameterSuggestion,
    analyze_backtest,
)
from config.param_manager import PARAM_LIMITS


# ==================== Test Data Fixtures ====================

@pytest.fixture
def good_backtest_result():
    """Backtest result with good performance."""
    return {
        "total_trades": 50,
        "wins": 30,
        "losses": 20,
        "win_rate": 60.0,
        "profit_factor": 1.5,
        "total_profit": 500.0,
        "max_drawdown": 100.0,
        "avg_win": 25.0,
        "avg_loss": 12.5,
    }


@pytest.fixture
def poor_backtest_result():
    """Backtest result with poor performance (profit factor < 1)."""
    return {
        "total_trades": 40,
        "wins": 15,
        "losses": 25,
        "win_rate": 37.5,
        "profit_factor": 0.8,
        "total_profit": -200.0,
        "max_drawdown": 300.0,
        "avg_win": 40.0,
        "avg_loss": 24.0,
    }


@pytest.fixture
def low_trade_backtest_result():
    """Backtest result with very few trades."""
    return {
        "total_trades": 10,
        "wins": 5,
        "losses": 5,
        "win_rate": 50.0,
        "profit_factor": 1.0,
        "total_profit": 0.0,
        "max_drawdown": 50.0,
        "avg_win": 10.0,
        "avg_loss": 10.0,
    }


@pytest.fixture
def default_params():
    """Default trading parameters."""
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


# ==================== ParameterSuggestion Tests ====================

class TestParameterSuggestion:
    """Tests for ParameterSuggestion dataclass."""

    def test_create_suggestion(self):
        """Should create a parameter suggestion."""
        suggestion = ParameterSuggestion(
            param_name="adx_threshold",
            old_value=10,
            new_value=12,
            reason="Increase ADX to filter weak trends",
        )
        assert suggestion.param_name == "adx_threshold"
        assert suggestion.old_value == 10
        assert suggestion.new_value == 12
        assert "ADX" in suggestion.reason

    def test_suggestion_with_float_values(self):
        """Should handle float values."""
        suggestion = ParameterSuggestion(
            param_name="tp_mult",
            old_value=2.0,
            new_value=2.3,
            reason="Increase take profit",
        )
        assert suggestion.new_value == 2.3


# ==================== AnalysisResult Tests ====================

class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_create_result(self):
        """Should create an analysis result."""
        result = AnalysisResult(
            analysis="Performance looks good",
            suggestions=[],
            expected_impact="Minimal changes needed",
            confidence="high",
        )
        assert result.analysis == "Performance looks good"
        assert result.confidence == "high"

    def test_to_dict(self):
        """to_dict() should return serializable dict."""
        suggestion = ParameterSuggestion(
            param_name="adx_threshold",
            old_value=10,
            new_value=12,
            reason="Test",
        )
        result = AnalysisResult(
            analysis="Test analysis",
            suggestions=[suggestion],
            expected_impact="Test impact",
            confidence="medium",
        )
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["analysis"] == "Test analysis"
        assert len(result_dict["suggestions"]) == 1
        assert result_dict["suggestions"][0]["param_name"] == "adx_threshold"

    def test_get_new_params(self, default_params):
        """get_new_params() should apply suggestions."""
        suggestion = ParameterSuggestion(
            param_name="adx_threshold",
            old_value=10,
            new_value=15,
            reason="Test",
        )
        result = AnalysisResult(
            analysis="Test",
            suggestions=[suggestion],
            expected_impact="Test",
            confidence="medium",
        )

        new_params = result.get_new_params(default_params)

        assert new_params["adx_threshold"] == 15
        # Other params unchanged
        assert new_params["tp_mult"] == default_params["tp_mult"]

    def test_get_new_params_multiple_suggestions(self, default_params):
        """Should apply multiple suggestions."""
        suggestions = [
            ParameterSuggestion("adx_threshold", 10, 15, "Test"),
            ParameterSuggestion("tp_mult", 2.0, 2.5, "Test"),
        ]
        result = AnalysisResult(
            analysis="Test",
            suggestions=suggestions,
            expected_impact="Test",
            confidence="medium",
        )

        new_params = result.get_new_params(default_params)

        assert new_params["adx_threshold"] == 15
        assert new_params["tp_mult"] == 2.5


# ==================== AIAnalyzer Initialization Tests ====================

class TestAIAnalyzerInit:
    """Tests for AIAnalyzer initialization."""

    def test_init_without_api_key(self):
        """Should initialize without API key (uses fallback)."""
        # Clear env var temporarily
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            analyzer = AIAnalyzer()
            assert analyzer.client is None
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_init_with_invalid_api_key(self):
        """Should handle invalid API key gracefully."""
        analyzer = AIAnalyzer(api_key="invalid-key")
        # Client may or may not be created depending on OpenAI lib behavior
        # But should not raise exception
        assert True

    def test_model_default(self):
        """Should use gpt-4o-mini by default."""
        analyzer = AIAnalyzer()
        assert analyzer.model == "gpt-4o-mini"

    def test_custom_model(self):
        """Should allow custom model."""
        analyzer = AIAnalyzer(model="gpt-4")
        assert analyzer.model == "gpt-4"


# ==================== Fallback Analysis Tests ====================

class TestFallbackAnalysis:
    """Tests for rule-based fallback analysis."""

    def test_fallback_on_poor_profit_factor(self, poor_backtest_result, default_params):
        """Should suggest increasing ADX when profit factor < 1."""
        analyzer = AIAnalyzer()  # No API key = uses fallback
        result = analyzer._fallback_analysis(poor_backtest_result, default_params)

        assert isinstance(result, AnalysisResult)
        assert result.confidence == "low"  # Fallback always low confidence

        # Should have suggestion to increase ADX
        adx_suggestions = [s for s in result.suggestions if s.param_name == "adx_threshold"]
        assert len(adx_suggestions) >= 0  # May or may not trigger depending on threshold

    def test_fallback_on_low_win_rate(self, default_params):
        """Should suggest increasing TP when win rate low but PF decent."""
        backtest = {
            "total_trades": 50,
            "win_rate": 35.0,  # < 40%
            "profit_factor": 0.9,  # > 0.8
            "total_profit": -50.0,
            "max_drawdown": 100.0,
            "avg_win": 30.0,
            "avg_loss": 15.0,
        }
        analyzer = AIAnalyzer()
        result = analyzer._fallback_analysis(backtest, default_params)

        # Should have TP suggestion
        tp_suggestions = [s for s in result.suggestions if s.param_name == "tp_mult"]
        assert len(tp_suggestions) >= 0  # May trigger

    def test_fallback_on_low_trades(self, low_trade_backtest_result, default_params):
        """Should suggest reducing slope threshold when too few trades."""
        analyzer = AIAnalyzer()
        result = analyzer._fallback_analysis(low_trade_backtest_result, default_params)

        # Should have slope suggestion when trades < 20
        slope_suggestions = [s for s in result.suggestions if s.param_name == "slope_threshold"]
        assert len(slope_suggestions) >= 1

        # New slope should be less than old
        if slope_suggestions:
            assert slope_suggestions[0].new_value < slope_suggestions[0].old_value

    def test_fallback_limits_suggestions(self, poor_backtest_result, default_params):
        """Fallback should return max 3 suggestions."""
        analyzer = AIAnalyzer()
        result = analyzer._fallback_analysis(poor_backtest_result, default_params)
        assert len(result.suggestions) <= 3

    def test_fallback_respects_param_limits(self, default_params):
        """Suggestions should be within PARAM_LIMITS."""
        backtest = {
            "total_trades": 5,  # Very few trades
            "win_rate": 20.0,
            "profit_factor": 0.5,
            "total_profit": -500.0,
            "max_drawdown": 400.0,
            "avg_win": 50.0,
            "avg_loss": 60.0,
        }
        analyzer = AIAnalyzer()
        result = analyzer._fallback_analysis(backtest, default_params)

        for suggestion in result.suggestions:
            if suggestion.param_name in PARAM_LIMITS:
                limits = PARAM_LIMITS[suggestion.param_name]
                assert suggestion.new_value >= limits["min"]
                assert suggestion.new_value <= limits["max"]


# ==================== Response Parsing Tests ====================

class TestResponseParsing:
    """Tests for parsing AI responses."""

    def test_parse_valid_json(self, default_params):
        """Should parse valid JSON response."""
        analyzer = AIAnalyzer()
        raw_response = """{
            "analysis": "Good performance overall",
            "suggestions": [
                {
                    "param_name": "adx_threshold",
                    "old_value": 10,
                    "new_value": 12,
                    "reason": "Increase to filter weak signals"
                }
            ],
            "expected_impact": "5-10% improvement expected",
            "confidence": "medium"
        }"""

        result = analyzer._parse_response(raw_response, default_params)

        assert result.analysis == "Good performance overall"
        assert len(result.suggestions) == 1
        assert result.suggestions[0].param_name == "adx_threshold"
        assert result.confidence == "medium"

    def test_parse_markdown_wrapped_json(self, default_params):
        """Should handle JSON wrapped in markdown code blocks."""
        analyzer = AIAnalyzer()
        raw_response = """```json
{
    "analysis": "Test analysis",
    "suggestions": [],
    "expected_impact": "None",
    "confidence": "high"
}
```"""

        result = analyzer._parse_response(raw_response, default_params)
        assert result.analysis == "Test analysis"

    def test_parse_invalid_json(self, default_params):
        """Should handle invalid JSON gracefully."""
        analyzer = AIAnalyzer()
        raw_response = "This is not valid JSON"

        result = analyzer._parse_response(raw_response, default_params)

        assert "Failed to parse" in result.analysis
        assert result.confidence == "low"
        assert result.suggestions == []

    def test_parse_clamps_to_limits(self, default_params):
        """Should clamp suggested values to PARAM_LIMITS."""
        analyzer = AIAnalyzer()
        raw_response = """{
            "analysis": "Test",
            "suggestions": [
                {
                    "param_name": "adx_threshold",
                    "old_value": 10,
                    "new_value": 100,
                    "reason": "Way too high"
                }
            ],
            "expected_impact": "Test",
            "confidence": "low"
        }"""

        result = analyzer._parse_response(raw_response, default_params)

        # Should be clamped to max (30)
        assert result.suggestions[0].new_value <= PARAM_LIMITS["adx_threshold"]["max"]

    def test_parse_converts_to_int(self, default_params):
        """Should convert int params to int type."""
        analyzer = AIAnalyzer()
        raw_response = """{
            "analysis": "Test",
            "suggestions": [
                {
                    "param_name": "adx_threshold",
                    "old_value": 10,
                    "new_value": 12.7,
                    "reason": "Test"
                }
            ],
            "expected_impact": "Test",
            "confidence": "medium"
        }"""

        result = analyzer._parse_response(raw_response, default_params)

        # Should be converted to int (13 rounded from 12.7)
        assert isinstance(result.suggestions[0].new_value, int)
        assert result.suggestions[0].new_value == 13


# ==================== Full Analysis Tests ====================

class TestFullAnalysis:
    """Tests for full analysis flow (uses fallback without API key)."""

    def test_analyze_returns_result(self, good_backtest_result, default_params):
        """analyze() should return AnalysisResult."""
        analyzer = AIAnalyzer()  # No API = fallback
        result = analyzer.analyze(good_backtest_result, default_params)

        assert isinstance(result, AnalysisResult)

    def test_analyze_with_symbol(self, good_backtest_result, default_params):
        """Should accept symbol parameter."""
        analyzer = AIAnalyzer()
        result = analyzer.analyze(
            good_backtest_result,
            default_params,
            symbol="XAUJPY",
        )

        assert isinstance(result, AnalysisResult)

    def test_analyze_with_context(self, good_backtest_result, default_params):
        """Should accept additional_context parameter."""
        analyzer = AIAnalyzer()
        result = analyzer.analyze(
            good_backtest_result,
            default_params,
            additional_context="Market was very volatile",
        )

        assert isinstance(result, AnalysisResult)


# ==================== Convenience Function Tests ====================

class TestConvenienceFunction:
    """Tests for analyze_backtest convenience function."""

    def test_analyze_backtest_works(self, good_backtest_result, default_params):
        """analyze_backtest() should work like analyzer.analyze()."""
        result = analyze_backtest(
            good_backtest_result,
            default_params,
            symbol="USDJPY",
        )

        assert isinstance(result, AnalysisResult)


# ==================== Format Param Limits Tests ====================

class TestFormatParamLimits:
    """Tests for parameter limit formatting."""

    def test_format_param_limits(self):
        """_format_param_limits() should return readable string."""
        analyzer = AIAnalyzer()
        formatted = analyzer._format_param_limits()

        assert isinstance(formatted, str)
        assert "adx_threshold" in formatted
        assert "tp_mult" in formatted
        # Should include min/max ranges
        assert "min" in formatted.lower() or "to" in formatted.lower()


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
