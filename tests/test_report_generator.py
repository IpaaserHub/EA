"""
Tests for AI Optimization Report Generator
============================================
Run with: .venv/bin/python -m pytest tests/test_report_generator.py -v

Tests for:
1. Fallback report generation (no API key needed)
2. Report structure and legal compliance
3. Analysis fallback
4. Disclaimer always present
"""

import pytest
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimizer.report_generator import ReportGenerator, DISCLAIMER_JP


# ==================== Test Fixtures ====================

def make_optimization_run_dict(applied=True, reason="Improvement detected"):
    """Create a mock OptimizationRun.to_dict() output."""
    return {
        "symbol": "USDJPY",
        "old_params": {
            "adx_threshold": 5,
            "slope_threshold": 0.00001,
            "buy_position": 0.50,
            "sell_position": 0.50,
            "tp_mult": 2.0,
            "sl_mult": 1.5,
        },
        "new_params": {
            "adx_threshold": 8,
            "slope_threshold": 0.00002,
            "buy_position": 0.48,
            "sell_position": 0.52,
            "tp_mult": 2.2,
            "sl_mult": 1.5,
        },
        "old_result": {
            "total_trades": 120,
            "wins": 60,
            "losses": 60,
            "win_rate": 50.0,
            "profit_factor": 1.1,
            "total_profit": 150.5,
            "gross_profit": 1650.0,
            "gross_loss": 1500.0,
            "max_drawdown": 200.0,
            "avg_win": 27.5,
            "avg_loss": 25.0,
        },
        "new_result": {
            "total_trades": 95,
            "wins": 55,
            "losses": 40,
            "win_rate": 57.89,
            "profit_factor": 1.45,
            "total_profit": 320.0,
            "gross_profit": 1400.0,
            "gross_loss": 965.5,
            "max_drawdown": 150.0,
            "avg_win": 25.45,
            "avg_loss": 24.14,
        },
        "improvement_pct": 15.5,
        "applied": applied,
        "reason": reason,
        "timestamp": "2026-02-06T10:00:00",
    }


def make_run_with_wfo():
    """Create a run dict with walk-forward results."""
    run = make_optimization_run_dict()
    run["walk_forward"] = {
        "robustness_ratio": 0.72,
        "is_robust": True,
        "avg_in_sample_pf": 1.65,
        "avg_out_sample_pf": 1.32,
        "n_splits": 5,
    }
    return run


# ==================== Report Generation Tests ====================

class TestFallbackReport:
    """Tests for fallback report (no LLM)."""

    def test_generates_report_without_api_key(self):
        """Should generate a report even without OpenAI API key."""
        generator = ReportGenerator(api_key="")
        run_data = make_optimization_run_dict()
        report = generator.generate(run_data)
        assert len(report) > 100

    def test_report_contains_symbol(self):
        """Report should mention the symbol."""
        generator = ReportGenerator(api_key="")
        run_data = make_optimization_run_dict()
        report = generator.generate(run_data)
        assert "USDJPY" in report

    def test_report_contains_disclaimer(self):
        """Report must ALWAYS contain the legal disclaimer."""
        generator = ReportGenerator(api_key="")
        run_data = make_optimization_run_dict()
        report = generator.generate(run_data)
        assert "免責事項" in report
        assert "将来の利益を保証するものではありません" in report

    def test_report_contains_performance_table(self):
        """Report should have performance comparison."""
        generator = ReportGenerator(api_key="")
        run_data = make_optimization_run_dict()
        report = generator.generate(run_data)
        assert "勝率" in report
        assert "プロフィットファクター" in report

    def test_report_shows_parameter_changes(self):
        """Report should list parameter changes."""
        generator = ReportGenerator(api_key="")
        run_data = make_optimization_run_dict()
        report = generator.generate(run_data)
        assert "adx_threshold" in report
        assert "5" in report  # old value
        assert "8" in report  # new value

    def test_report_with_wfo(self):
        """Report should include walk-forward section when available."""
        generator = ReportGenerator(api_key="")
        run_data = make_run_with_wfo()
        report = generator.generate(run_data)
        assert "ウォークフォワード" in report
        assert "ロバストネス" in report

    def test_report_not_applied(self):
        """Report should indicate when params were NOT applied."""
        generator = ReportGenerator(api_key="")
        run_data = make_optimization_run_dict(applied=False, reason="Drawdown too high")
        report = generator.generate(run_data)
        assert "更新されませんでした" in report

    def test_report_no_changes(self):
        """Report should handle case where no params changed."""
        generator = ReportGenerator(api_key="")
        run_data = make_optimization_run_dict()
        run_data["new_params"] = run_data["old_params"].copy()
        report = generator.generate(run_data)
        assert "パラメータ変更なし" in report


# ==================== Analysis Fallback Tests ====================

class TestFallbackAnalysis:
    """Tests for fallback analysis (no LLM)."""

    def test_fallback_analysis_is_valid_json(self):
        """Fallback analysis should produce valid JSON."""
        generator = ReportGenerator(api_key="")
        run_data = make_optimization_run_dict()
        analysis = generator._fallback_analysis(run_data)
        parsed = json.loads(analysis)
        assert "summary" in parsed
        assert "performance_comparison" in parsed
        assert "overall_rating" in parsed

    def test_fallback_analysis_detects_improvement(self):
        """Should detect improvement in profit factor."""
        generator = ReportGenerator(api_key="")
        run_data = make_optimization_run_dict()
        analysis = json.loads(generator._fallback_analysis(run_data))
        assert analysis["overall_rating"] == "improved"

    def test_fallback_analysis_detects_degradation(self):
        """Should detect degradation in profit factor."""
        generator = ReportGenerator(api_key="")
        run_data = make_optimization_run_dict()
        # Swap old and new so new is worse
        run_data["old_result"], run_data["new_result"] = (
            run_data["new_result"], run_data["old_result"]
        )
        analysis = json.loads(generator._fallback_analysis(run_data))
        assert analysis["overall_rating"] == "degraded"

    def test_fallback_analysis_lists_changes(self):
        """Should list parameter changes."""
        generator = ReportGenerator(api_key="")
        run_data = make_optimization_run_dict()
        analysis = json.loads(generator._fallback_analysis(run_data))
        assert len(analysis["key_changes"]) > 0

    def test_fallback_analysis_with_wfo(self):
        """Should include walk-forward info."""
        generator = ReportGenerator(api_key="")
        run_data = make_run_with_wfo()
        analysis = json.loads(generator._fallback_analysis(run_data))
        assert "Robustness" in analysis["walk_forward_summary"]


# ==================== Disclaimer Tests ====================

class TestDisclaimer:
    """Tests for legal compliance."""

    def test_disclaimer_never_empty(self):
        """Disclaimer constant should never be empty."""
        assert len(DISCLAIMER_JP) > 50

    def test_disclaimer_has_required_phrases(self):
        """Disclaimer must contain required legal phrases."""
        assert "過去のデータに基づく" in DISCLAIMER_JP
        assert "将来の利益を保証するものではありません" in DISCLAIMER_JP
        assert "免責事項" in DISCLAIMER_JP

    def test_disclaimer_always_appended(self):
        """Disclaimer must be at the end of every report."""
        generator = ReportGenerator(api_key="")
        run_data = make_optimization_run_dict()
        report = generator.generate(run_data)
        # Disclaimer should be at the end
        assert report.strip().endswith(DISCLAIMER_JP.strip())


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
