"""
Action Plan & Report Agent
Model: GPT-5 (MCP tool, same rationale as Impact Agent)

Tools: generate_final_report, create_action_roadmap, generate_suggested_fixes,
       prepare_apply_fix_payload
it produces the investigation report and apply-fix payloads shown in the diagram's
"suggested fixes" / "Report" / "apply fixes" outputs.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Optional, Callable
from schemas import InvestigationState


class ActionPlanReportAgent:
    role_name = "action_plan_report_agent"
    model = "gpt-5"

    def __init__(self, mcp_gpt5_caller: Optional[Callable] = None):
        self.mcp_gpt5_caller = mcp_gpt5_caller

    def create_action_roadmap(self, state: InvestigationState) -> list[dict]:
        roadmap = []
        for i, rec in enumerate(state.recommendations, start=1):
            matching_impact = next(
                (item for item in state.impact_summary.get("items", [])
                 if item.get("title") == rec.title),
                {},
            )
            roadmap.append({
                "priority": i,
                "title": rec.title,
                "target_activity": rec.target_activity,
                "description": rec.description,
                "effort": matching_impact.get("implementation_effort", "unknown"),
                "risk": matching_impact.get("risk_level", "unknown"),
                "estimated_gain_pct": matching_impact.get("estimated_performance_gain_pct"),
            })
        return roadmap

    def generate_suggested_fixes(self, state: InvestigationState) -> list[dict]:
        fixes = []
        for rec in state.recommendations:
            fixes.append({
                "target_activity": rec.target_activity,
                "fix_title": rec.title,
                "fix_description": rec.description,
                "confidence": rec.confidence,
            })
        return fixes

    def prepare_apply_fix_payload(self, fix: dict, ctx) -> dict:
        return {
            "subscription_id": ctx.subscription_id,
            "service": ctx.service,
            "resource_group": ctx.resource_group,
            "workspace_name": ctx.workspace_name,
            "item_name": ctx.item_name,
            "target_activity": fix.get("target_activity"),
            "fix_title": fix.get("fix_title"),
            "status": "pending_approval",
        }

    async def generate_final_report(self, state: InvestigationState) -> str:
        ctx = state.context
        payload = {
            "service": ctx.service,
            "item_name": ctx.item_name,
            "pipeline_health": ctx.pipeline_health,
            "pipeline_deviation_pct": ctx.pipeline_deviation_pct,
            "root_causes": [rc.description for rc in state.root_causes],
            "recommendations": [r.title for r in state.recommendations],
            "impact_summary": state.impact_summary.get("summary"),
        }
        narrative = (
            f"Investigation for {ctx.item_name} ({ctx.service}) found "
            f"{len(state.root_causes)} root cause(s) and {len(state.recommendations)} "
            f"recommendation(s). {state.impact_summary.get('summary', '')}"
        )
        if self.mcp_gpt5_caller is not None:
            try:
                narrative = await self.mcp_gpt5_caller(
                    system_prompt="Write a clear executive-summary narrative for an AIOps RCA report.",
                    user_prompt=json.dumps(payload),
                )
            except Exception:
                pass
        return narrative

    async def generate(self, state: InvestigationState) -> dict:
        narrative = await self.generate_final_report(state)
        roadmap = self.create_action_roadmap(state)
        fixes = self.generate_suggested_fixes(state)
        ctx = state.context
        apply_fix_payloads = [self.prepare_apply_fix_payload(f, ctx) for f in fixes]

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "subscription_id": ctx.subscription_id,
            "service": ctx.service,
            "resource_group": ctx.resource_group,
            "workspace_name": ctx.workspace_name,
            "item_name": ctx.item_name,
            "executive_summary": narrative,
            "pipeline_health": ctx.pipeline_health,
            "pipeline_deviation_pct": ctx.pipeline_deviation_pct,
            "root_causes": [
                {"category": rc.category, "description": rc.description, "confidence": rc.confidence}
                for rc in state.root_causes
            ],
            "action_roadmap": roadmap,
            "suggested_fixes": fixes,
            "apply_fix_payloads": apply_fix_payloads,
        }