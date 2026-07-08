"""
Impact Agent
Model: GPT-5 (Azure OpenAI deployment, via mcp_gpt5_caller) when configured,
otherwise a deterministic scoring pass over the recommendations.

Tools: estimate_performance_gain, estimate_implementation_effort,
       assess_risk_level, generate_impact_summary
"""
from __future__ import annotations
import json
from typing import Optional, Callable
from schemas import InvestigationContext, Recommendation

import azure_openai_client


class ImpactAgent:
    role_name = "impact_agent"

    def __init__(self, mcp_gpt5_caller: Optional[Callable] = None):
        # mcp_gpt5_caller(system_prompt, user_prompt) -> str
        # Pluggable so the actual transport (Azure OpenAI today, MCP later) lives
        # outside this class - see azure_openai_client.azure_gpt5_caller.
        self.mcp_gpt5_caller = mcp_gpt5_caller
        self._last_call_meta: dict = {}
        self._last_provider: str = "deterministic"

    @property
    def model(self) -> str:
        deployment = (self._last_call_meta or {}).get("deployment")
        if self._last_provider == "azure_openai" and deployment:
            return deployment
        return "deterministic"

    def estimate_performance_gain(self, rec: Recommendation) -> float:
        return round(rec.impact_score * 100, 1)  

    def estimate_implementation_effort(self, rec: Recommendation) -> str:
        if rec.effort_score < 0.3:
            return "low"
        if rec.effort_score < 0.6:
            return "medium"
        return "high"

    def assess_risk_level(self, rec: Recommendation) -> str:
        if rec.risk_score < 0.3:
            return "low"
        if rec.risk_score < 0.6:
            return "medium"
        return "high"

    async def assess(self, ctx: InvestigationContext, recommendations: list[Recommendation]) -> dict:
        self._last_call_meta = {}
        self._last_provider = "deterministic"

        if not recommendations:
            return {"summary": "No recommendations to assess.", "detailed_summary": None, "items": []}

        items = []
        for rec in recommendations:
            items.append({
                "title": rec.title,
                "target_activity": rec.target_activity,
                "estimated_performance_gain_pct": self.estimate_performance_gain(rec),
                "implementation_effort": self.estimate_implementation_effort(rec),
                "risk_level": self.assess_risk_level(rec),
            })

        # `summary` is ALWAYS this short, deterministic one-liner - this is
        # what dashboards should show as the headline (e.g. "Estimated ~78%
        # processing decrease..."). The optional GPT-5 narrative below is
        # exposed separately as `detailed_summary` for an expandable view,
        # rather than overwriting the headline with a multi-paragraph essay.
        headline = self.generate_impact_summary(ctx, items)
        detailed_summary = None
        summary_error = None
        if self.mcp_gpt5_caller is not None:
            try:
                detailed_summary = await self.mcp_gpt5_caller(
                    system_prompt=(
                        "You write a concise operational/financial/performance "
                        "impact summary for AIOps remediation recommendations."
                    ),
                    user_prompt=json.dumps({"service": ctx.service, "item_name": ctx.item_name, "items": items}),
                )
                self._last_provider = "azure_openai"
                self._last_call_meta = dict(azure_openai_client.last_call_meta)
            except Exception as e:
                # Fall back to the deterministic summary rather than failing
                # the whole investigation over an impact-narrative call, but
                # record the failure instead of silently swallowing it.
                summary_error = f"{type(e).__name__}: {e}"
                self._last_provider = "deterministic_fallback"
                self._last_call_meta = {"error": summary_error}

        result = {"summary": headline, "detailed_summary": detailed_summary, "items": items}
        if summary_error:
            result["summary_generation_error"] = summary_error
        return result

    def generate_impact_summary(self, ctx: InvestigationContext, items: list[dict]) -> str:
        if not items:
            return "No actionable items."
        best = max(items, key=lambda i: i["estimated_performance_gain_pct"])
        return (
            f"{len(items)} recommendation(s) identified for {ctx.item_name} ({ctx.service}). "
            f"Highest-impact item: '{best['title']}' (~{best['estimated_performance_gain_pct']}% "
            f"estimated gain, {best['implementation_effort']} effort, {best['risk_level']} risk)."
        )