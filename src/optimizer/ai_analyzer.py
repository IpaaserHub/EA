"""
AI Analyzer
===========
Uses OpenAI to analyze backtest results and suggest parameter improvements.

This module:
1. Formats backtest results into a structured prompt
2. Sends to OpenAI for analysis
3. Parses the response into parameter suggestions
4. Validates suggestions are within safe limits
"""

import json
import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Import param limits from config module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.param_manager import PARAM_LIMITS

logger = logging.getLogger(__name__)


@dataclass
class ParameterSuggestion:
    """A single parameter change suggestion."""
    param_name: str
    old_value: Any
    new_value: Any
    reason: str


@dataclass
class AnalysisResult:
    """Result from AI analysis."""
    analysis: str
    suggestions: List[ParameterSuggestion]
    expected_impact: str
    confidence: str  # "high", "medium", "low"
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analysis": self.analysis,
            "suggestions": [
                {
                    "param_name": s.param_name,
                    "old_value": s.old_value,
                    "new_value": s.new_value,
                    "reason": s.reason,
                }
                for s in self.suggestions
            ],
            "expected_impact": self.expected_impact,
            "confidence": self.confidence,
        }

    def get_new_params(self, current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply suggestions to current params."""
        new_params = current_params.copy()
        for suggestion in self.suggestions:
            new_params[suggestion.param_name] = suggestion.new_value
        return new_params


class AIAnalyzer:
    """
    Analyzes backtest results using OpenAI.

    Usage:
        analyzer = AIAnalyzer(api_key="sk-...")
        result = analyzer.analyze(backtest_result, current_params)
        new_params = result.get_new_params(current_params)
    """

    SYSTEM_PROMPT = """You are a trading strategy optimization expert. Your task is to analyze backtest results and suggest parameter improvements.

RULES:
1. Only suggest small, incremental changes (max 20% change per parameter)
2. Never suggest values outside the allowed ranges
3. Focus on improving profit factor and reducing drawdown
4. Explain your reasoning clearly
5. Be conservative - it's better to make small improvements than break a working strategy

PARAMETER RANGES:
{param_limits}

OUTPUT FORMAT (JSON):
{{
    "analysis": "Brief analysis of current performance",
    "suggestions": [
        {{
            "param_name": "parameter_name",
            "old_value": current_value,
            "new_value": suggested_value,
            "reason": "Why this change will help"
        }}
    ],
    "expected_impact": "What improvement to expect",
    "confidence": "high/medium/low"
}}

Only output valid JSON. No markdown, no explanations outside the JSON."""

    USER_PROMPT_TEMPLATE = """Analyze this backtest result and suggest parameter improvements:

SYMBOL: {symbol}
CURRENT PARAMETERS:
{current_params}

BACKTEST RESULTS:
- Total Trades: {total_trades}
- Win Rate: {win_rate}%
- Profit Factor: {profit_factor}
- Total Profit: {total_profit}
- Max Drawdown: {max_drawdown}
- Average Win: {avg_win}
- Average Loss: {avg_loss}

{additional_context}

Suggest up to 3 parameter changes to improve performance. Output JSON only."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize AI analyzer.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (default: gpt-4o-mini for cost efficiency)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = None

        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("OpenAI package not installed")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")

    def analyze(
        self,
        backtest_result: Dict[str, Any],
        current_params: Dict[str, Any],
        symbol: str = "UNKNOWN",
        additional_context: str = "",
    ) -> AnalysisResult:
        """
        Analyze backtest results and get parameter suggestions.

        Args:
            backtest_result: Dict with backtest metrics
            current_params: Current parameter values
            symbol: Trading symbol
            additional_context: Extra info for the AI

        Returns:
            AnalysisResult with suggestions
        """
        if not self.client:
            logger.warning("OpenAI client not available, using fallback analysis")
            return self._fallback_analysis(backtest_result, current_params)

        # Build prompts
        system_prompt = self.SYSTEM_PROMPT.format(
            param_limits=self._format_param_limits()
        )

        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            symbol=symbol,
            current_params=json.dumps(current_params, indent=2),
            total_trades=backtest_result.get("total_trades", 0),
            win_rate=backtest_result.get("win_rate", 0),
            profit_factor=backtest_result.get("profit_factor", 0),
            total_profit=backtest_result.get("total_profit", 0),
            max_drawdown=backtest_result.get("max_drawdown", 0),
            avg_win=backtest_result.get("avg_win", 0),
            avg_loss=backtest_result.get("avg_loss", 0),
            additional_context=additional_context,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Lower temperature for more consistent suggestions
                max_tokens=1000,
            )

            raw_response = response.choices[0].message.content
            return self._parse_response(raw_response, current_params)

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._fallback_analysis(backtest_result, current_params)

    def _parse_response(
        self, raw_response: str, current_params: Dict[str, Any]
    ) -> AnalysisResult:
        """Parse AI response into AnalysisResult."""
        try:
            # Clean up response (remove markdown if present)
            clean_response = raw_response.strip()
            if clean_response.startswith("```"):
                clean_response = clean_response.split("```")[1]
                if clean_response.startswith("json"):
                    clean_response = clean_response[4:]
            clean_response = clean_response.strip()

            data = json.loads(clean_response)

            suggestions = []
            for s in data.get("suggestions", []):
                # Validate suggestion is within limits
                param_name = s.get("param_name")
                new_value = s.get("new_value")

                if param_name in PARAM_LIMITS:
                    limits = PARAM_LIMITS[param_name]
                    # Clamp to limits
                    new_value = max(limits["min"], min(limits["max"], new_value))
                    # Convert type
                    if limits["type"] == "int":
                        new_value = int(round(new_value))

                suggestions.append(
                    ParameterSuggestion(
                        param_name=param_name,
                        old_value=s.get("old_value", current_params.get(param_name)),
                        new_value=new_value,
                        reason=s.get("reason", ""),
                    )
                )

            return AnalysisResult(
                analysis=data.get("analysis", ""),
                suggestions=suggestions,
                expected_impact=data.get("expected_impact", ""),
                confidence=data.get("confidence", "medium"),
                raw_response=raw_response,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            return AnalysisResult(
                analysis="Failed to parse AI response",
                suggestions=[],
                expected_impact="Unknown",
                confidence="low",
                raw_response=raw_response,
            )

    def _fallback_analysis(
        self, backtest_result: Dict[str, Any], current_params: Dict[str, Any]
    ) -> AnalysisResult:
        """
        Simple rule-based analysis when AI is not available.

        This provides basic suggestions without requiring OpenAI.
        """
        suggestions = []
        analysis_parts = []

        profit_factor = backtest_result.get("profit_factor", 0)
        win_rate = backtest_result.get("win_rate", 0)
        total_trades = backtest_result.get("total_trades", 0)

        # Rule 1: If profit factor < 1, increase ADX threshold (filter weak trends)
        if profit_factor < 1.0:
            current_adx = current_params.get("adx_threshold", 5)
            new_adx = min(current_adx + 2, PARAM_LIMITS["adx_threshold"]["max"])
            if new_adx != current_adx:
                suggestions.append(
                    ParameterSuggestion(
                        param_name="adx_threshold",
                        old_value=current_adx,
                        new_value=new_adx,
                        reason="Filter weak trend trades by requiring stronger ADX",
                    )
                )
                analysis_parts.append("Profit factor below 1.0 suggests too many losing trades")

        # Rule 2: If win rate < 40% and profit factor > 0.8, increase TP multiplier
        if win_rate < 40 and profit_factor > 0.8:
            current_tp = current_params.get("tp_mult", 2.0)
            new_tp = min(current_tp + 0.3, PARAM_LIMITS["tp_mult"]["max"])
            if new_tp != current_tp:
                suggestions.append(
                    ParameterSuggestion(
                        param_name="tp_mult",
                        old_value=current_tp,
                        new_value=round(new_tp, 1),
                        reason="Increase take profit to capture larger moves",
                    )
                )
                analysis_parts.append("Low win rate but decent profit factor suggests wins are too small")

        # Rule 3: If too few trades, reduce slope threshold
        if total_trades < 20:
            current_slope = current_params.get("slope_threshold", 0.00001)
            new_slope = max(current_slope * 0.8, PARAM_LIMITS["slope_threshold"]["min"])
            if new_slope != current_slope:
                suggestions.append(
                    ParameterSuggestion(
                        param_name="slope_threshold",
                        old_value=current_slope,
                        new_value=new_slope,
                        reason="Reduce slope threshold to generate more trading opportunities",
                    )
                )
                analysis_parts.append("Very few trades generated, parameters may be too strict")

        analysis = ". ".join(analysis_parts) if analysis_parts else "Parameters appear reasonable"

        return AnalysisResult(
            analysis=analysis,
            suggestions=suggestions[:3],  # Max 3 suggestions
            expected_impact="Small incremental improvement expected",
            confidence="low",  # Fallback analysis is less reliable
        )

    def _format_param_limits(self) -> str:
        """Format parameter limits for the prompt."""
        lines = []
        for name, limits in PARAM_LIMITS.items():
            lines.append(f"- {name}: {limits['min']} to {limits['max']} ({limits['type']})")
        return "\n".join(lines)


def analyze_backtest(
    backtest_result: Dict[str, Any],
    current_params: Dict[str, Any],
    symbol: str = "UNKNOWN",
    api_key: Optional[str] = None,
) -> AnalysisResult:
    """
    Convenience function to analyze backtest results.

    Args:
        backtest_result: Dict with backtest metrics
        current_params: Current parameter values
        symbol: Trading symbol
        api_key: OpenAI API key (optional)

    Returns:
        AnalysisResult with suggestions
    """
    analyzer = AIAnalyzer(api_key=api_key)
    return analyzer.analyze(backtest_result, current_params, symbol)
