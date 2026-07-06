"""
Impact Agent
Model: GPT-5 

Tools: estimate_performance_gain, estimate_implementation_effort,
       assess_risk_level, generate_impact_summary
"""
from __future__ import annotations
import json
from typing import Optional, Callable
from schemas import InvestigationContext, Recommendation


class ImpactAgent:
    role_name = "impact_agent"
    model = "gpt-5"  # invoked via MCP tool call, not claude_agent_sdk subagent

    def __init__(self, mcp_gpt5_caller: Optional[Callable] = None):
        # mcp_gpt5_caller(system_prompt, user_prompt) -> str
        # Pluggable so the actual MCP transport (server URL, auth) lives
        self.mcp_gpt5_caller = mcp_gpt5_caller

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
        if not recommendations:
            return {"summary": "No recommendations to assess.", "items": []}

        items = []
        for rec in recommendations:
            items.append({
                "title": rec.title,
                "target_activity": rec.target_activity,
                "estimated_performance_gain_pct": self.estimate_performance_gain(rec),
                "implementation_effort": self.estimate_implementation_effort(rec),
                "risk_level": self.assess_risk_level(rec),
            })

        summary_text = self.generate_impact_summary(ctx, items)
        if self.mcp_gpt5_caller is not None:
            try:
                summary_text = await self.mcp_gpt5_caller(
                    system_prompt=(
                        "You write a concise operational/financial/performance "
                        "impact summary for AIOps remediation recommendations."
                    ),
                    user_prompt=json.dumps({"service": ctx.service, "item_name": ctx.item_name, "items": items}),
                )
            except Exception:
                pass  

        return {"summary": summary_text, "items": items}

    def generate_impact_summary(self, ctx: InvestigationContext, items: list[dict]) -> str:
        if not items:
            return "No actionable items."
        best = max(items, key=lambda i: i["estimated_performance_gain_pct"])
        return (
            f"{len(items)} recommendation(s) identified for {ctx.item_name} ({ctx.service}). "
            f"Highest-impact item: '{best['title']}' (~{best['estimated_performance_gain_pct']}% "
            f"estimated gain, {best['implementation_effort']} effort, {best['risk_level']} risk)."
        )