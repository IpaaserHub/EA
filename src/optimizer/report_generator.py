"""
AI Optimization Report Generator
=================================
Generates human-readable Japanese optimization reports using LLM.

Two-pass LLM pattern:
1. Pass 1: Feed OptimizationRun data → get structured JSON analysis
2. Pass 2: Feed JSON analysis → get natural language Japanese report

Uses gpt-4o-mini (~0.2 yen per report).

Legal requirements (from business docs):
- Never say 「必ず儲かる」(guaranteed profit)
- Always include 「過去のデータに基づく結果」(based on historical data)
- Always include full 免責事項 (disclaimer)
- These are HARD-CODED in the template, not left to LLM
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Hard-coded legal disclaimer - NEVER left to LLM
DISCLAIMER_JP = """
【免責事項】
本レポートは過去のデータに基づく分析結果であり、将来の利益を保証するものではありません。
自動売買システムにはリスクが伴います。投資判断はご自身の責任で行ってください。
過去のパフォーマンスは将来の結果を示唆するものではありません。
""".strip()


class ReportGenerator:
    """
    Generates AI-powered optimization reports in Japanese.

    Usage:
        generator = ReportGenerator(api_key="sk-...")
        report = generator.generate(optimization_run_dict)
        print(report)
    """

    ANALYSIS_PROMPT = """You are a trading strategy analysis expert. Analyze the following optimization run data and produce a structured JSON analysis.

INPUT DATA:
{run_data}

Produce a JSON response with exactly these fields:
{{
    "summary": "1-2 sentence summary of what changed and why",
    "performance_comparison": {{
        "old_win_rate": number,
        "new_win_rate": number,
        "old_profit_factor": number,
        "new_profit_factor": number,
        "old_total_profit": number,
        "new_total_profit": number,
        "old_max_drawdown": number,
        "new_max_drawdown": number,
        "old_total_trades": number,
        "new_total_trades": number
    }},
    "key_changes": ["list of the most important parameter changes and their effects"],
    "risk_assessment": "brief risk assessment (1-2 sentences)",
    "walk_forward_summary": "walk-forward validation summary if available, otherwise 'N/A'",
    "overall_rating": "improved/stable/degraded"
}}

Only output valid JSON. No markdown."""

    REPORT_PROMPT = """あなたはFX自動売買システムの最適化レポートライターです。
以下のJSON分析データを基に、日本語の最適化レポートを作成してください。

分析データ:
{analysis_json}

シンボル: {symbol}
最適化日時: {timestamp}

以下のセクションを含むレポートを作成してください：

1. 📊 概要（3行以内で要約）
2. 📈 パフォーマンス比較（テーブル形式で新旧比較）
3. 🔧 パラメータ変更の説明（変更内容と理由）
4. 📉 ウォークフォワード検証結果（利用可能な場合）
5. ⚠️ リスク評価

重要なルール：
- 「必ず儲かる」などの断定的表現は絶対に使わないでください
- 客観的かつ慎重なトーンで書いてください
- 数値は小数点2桁まで表示してください
- マークダウン形式で書いてください

レポートのみを出力してください。"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize report generator.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (default gpt-4o-mini)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API. Returns response text."""
        if not self.api_key:
            logger.warning("No OpenAI API key — using fallback report")
            return ""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    def generate(self, run_data: Dict[str, Any]) -> str:
        """
        Generate a Japanese optimization report.

        Args:
            run_data: OptimizationRun.to_dict() output

        Returns:
            Formatted report string (markdown)
        """
        symbol = run_data.get("symbol", "Unknown")
        timestamp = run_data.get("timestamp", datetime.now().isoformat())

        # Pass 1: Get structured analysis
        analysis_json = self._get_analysis(run_data)

        # Pass 2: Generate Japanese report
        report = self._generate_report(analysis_json, symbol, timestamp)

        if not report:
            report = self._fallback_report(run_data)

        # Always append hard-coded disclaimer
        report += f"\n\n---\n\n{DISCLAIMER_JP}"

        return report

    def _get_analysis(self, run_data: Dict[str, Any]) -> str:
        """Pass 1: Get structured JSON analysis from LLM."""
        run_data_str = json.dumps(run_data, indent=2, ensure_ascii=False)

        response = self._call_llm(
            "You are a trading strategy analysis expert. Output valid JSON only.",
            self.ANALYSIS_PROMPT.format(run_data=run_data_str),
        )

        if not response:
            return self._fallback_analysis(run_data)

        # Validate it's JSON
        try:
            json.loads(response)
            return response
        except json.JSONDecodeError:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    json.loads(response[start:end])
                    return response[start:end]
                except json.JSONDecodeError:
                    pass
            return self._fallback_analysis(run_data)

    def _generate_report(self, analysis_json: str, symbol: str, timestamp: str) -> str:
        """Pass 2: Generate Japanese report from analysis."""
        response = self._call_llm(
            "あなたはFX自動売買システムの最適化レポートライターです。マークダウン形式で日本語のレポートを書いてください。",
            self.REPORT_PROMPT.format(
                analysis_json=analysis_json,
                symbol=symbol,
                timestamp=timestamp,
            ),
        )
        return response

    def _fallback_analysis(self, run_data: Dict[str, Any]) -> str:
        """Generate analysis without LLM."""
        old = run_data.get("old_result", {})
        new = run_data.get("new_result", {})

        analysis = {
            "summary": "Optimization completed. Parameters updated based on backtest results.",
            "performance_comparison": {
                "old_win_rate": old.get("win_rate", 0),
                "new_win_rate": new.get("win_rate", 0),
                "old_profit_factor": old.get("profit_factor", 0),
                "new_profit_factor": new.get("profit_factor", 0),
                "old_total_profit": old.get("total_profit", 0),
                "new_total_profit": new.get("total_profit", 0),
                "old_max_drawdown": old.get("max_drawdown", 0),
                "new_max_drawdown": new.get("max_drawdown", 0),
                "old_total_trades": old.get("total_trades", 0),
                "new_total_trades": new.get("total_trades", 0),
            },
            "key_changes": [],
            "risk_assessment": "Standard optimization run.",
            "walk_forward_summary": "N/A",
            "overall_rating": "stable",
        }

        # Determine rating
        old_pf = old.get("profit_factor", 0)
        new_pf = new.get("profit_factor", 0)
        if new_pf > old_pf * 1.05:
            analysis["overall_rating"] = "improved"
        elif new_pf < old_pf * 0.95:
            analysis["overall_rating"] = "degraded"

        # Key changes
        old_params = run_data.get("old_params", {})
        new_params = run_data.get("new_params", {})
        for key in new_params:
            if key in old_params and old_params[key] != new_params[key]:
                analysis["key_changes"].append(
                    f"{key}: {old_params[key]} → {new_params[key]}"
                )

        # Walk-forward
        wf = run_data.get("walk_forward")
        if wf:
            analysis["walk_forward_summary"] = (
                f"Robustness: {wf.get('robustness_ratio', 'N/A')}, "
                f"Robust: {wf.get('is_robust', 'N/A')}"
            )

        return json.dumps(analysis, indent=2, ensure_ascii=False)

    def _fallback_report(self, run_data: Dict[str, Any]) -> str:
        """Generate report without LLM (template-based)."""
        old = run_data.get("old_result", {})
        new = run_data.get("new_result", {})
        symbol = run_data.get("symbol", "Unknown")
        applied = run_data.get("applied", False)
        reason = run_data.get("reason", "")

        report = f"""# 最適化レポート: {symbol}

## 📊 概要

{symbol}の最適化を実行しました。
パラメータ{'を更新しました' if applied else 'は更新されませんでした'}。
{'理由: ' + reason if reason else ''}

## 📈 パフォーマンス比較

| 指標 | 変更前 | 変更後 |
|------|--------|--------|
| 勝率 | {old.get('win_rate', 0):.2f}% | {new.get('win_rate', 0):.2f}% |
| プロフィットファクター | {old.get('profit_factor', 0):.2f} | {new.get('profit_factor', 0):.2f} |
| 総利益 | {old.get('total_profit', 0):.2f} | {new.get('total_profit', 0):.2f} |
| 最大ドローダウン | {old.get('max_drawdown', 0):.2f} | {new.get('max_drawdown', 0):.2f} |
| 取引回数 | {old.get('total_trades', 0)} | {new.get('total_trades', 0)} |

## 🔧 パラメータ変更

"""
        old_params = run_data.get("old_params", {})
        new_params = run_data.get("new_params", {})
        changes = []
        for key in new_params:
            if key in old_params and old_params[key] != new_params[key]:
                changes.append(f"- **{key}**: {old_params[key]} → {new_params[key]}")

        if changes:
            report += "\n".join(changes)
        else:
            report += "パラメータ変更なし"

        wf = run_data.get("walk_forward")
        if wf:
            report += f"""

## 📉 ウォークフォワード検証結果

- ロバストネス比率: {wf.get('robustness_ratio', 'N/A')}
- 検証結果: {'合格 ✅' if wf.get('is_robust') else '不合格 ❌'}
"""

        report += """

## ⚠️ リスク評価

本結果は過去のデータに基づくバックテスト結果です。実際の取引結果とは異なる場合があります。"""

        return report
