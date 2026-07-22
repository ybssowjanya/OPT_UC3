"""
Action Plan & Report Agent
Model: GPT-5 (MCP tool, same rationale as Impact Agent)

Tools: generate_final_report, create_action_roadmap, generate_suggested_fixes,
       prepare_apply_fix_payload, generate_code_patches
it produces the investigation report and apply-fix payloads shown in the diagram's
"""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from typing import Optional, Callable
from schemas import InvestigationState
from keyvault_client import get_secret
import azure_openai_client
from sub_agents.base_agent import anthropic_ready
from calendar import monthrange
try:
    from claude_agent_sdk import ClaudeAgentOptions, query as sdk_query
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

try:
    import anthropic
except ImportError:
    anthropic = None

CODE_PATCH_MODEL = "claude-sonnet-4-6"

CODE_PATCH_SYSTEM = (
    "You are a senior Spark/PySpark performance engineer. You are given the "
    "FULL current source of one notebook activity plus the specific, "
    "evidence-backed findings/recommendations that target it. "
    "Rewrite the notebook so every cited issue is fixed, while preserving "
    "its behavior/output columns and NOT inventing unrelated changes. "
    "Return ONLY a JSON object, no markdown fences, no prose outside JSON, "
    "with exactly these keys:\n"
    '  "has_issues": true,\n'
    '  "patched_code": "<the full corrected notebook source as one string>",\n'
    '  "explanation": "<2-4 sentences, what changed and why, tied to the '
    'findings given>"\n'
    "If, after reviewing, you find the cited findings do not actually "
    "warrant a code change (e.g. they are purely infra/config issues with "
    "no code-side fix), return has_issues=false, patched_code=null, and "
    "explain why in \"explanation\"."
)


class ActionPlanReportAgent:
    role_name = "action_plan_report_agent"

    def __init__(self, mcp_gpt5_caller: Optional[Callable] = None):
        self.mcp_gpt5_caller = mcp_gpt5_caller
        self._last_call_meta: dict = {}
        self._last_provider: str = "deterministic"

    @property
    def model(self) -> str:
        deployment = (self._last_call_meta or {}).get("deployment")
        if self._last_provider == "azure_openai" and deployment:
            return deployment
        return "deterministic"

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

    # ---- compilation adjustment one-liner -----------------------------
    _PATCH_PHRASE_PATTERNS = [
        (r"inferSchema.*false|schema\(", "enforced explicit schema-on-read"),
        (r"\.cache\(\)|\.persist\(", "introduced DataFrame caching"),
        (r"unpersist\(\)", "released cached memory after use"),
        (r"df\.count\(\)", "removed redundant count() scans"),
        (r"shuffle\.partitions|shuffle partitions", "tuned shuffle partition sizing"),
        (r"import \*", "replaced wildcard imports with explicit imports"),
        (r"\.select\(|partition filter|predicate pushdown", "added column/partition pruning"),
        (r"display\(", "trimmed unnecessary preview scans"),
    ]

    def generate_compilation_adjustment(self, code_patches: dict) -> str:
        patched_count = sum(1 for p in code_patches.values() if p.get("has_issues"))
        if patched_count == 0:
            return "No code-level adjustments were required; findings were infrastructure/configuration related."

        haystack = "\n".join(
            f"{p.get('patched_code') or ''}\n{p.get('explanation') or ''}"
            for p in code_patches.values() if p.get("has_issues")
        )
        phrases: list[str] = []
        for pattern, phrase in self._PATCH_PHRASE_PATTERNS:
            if re.search(pattern, haystack, re.IGNORECASE) and phrase not in phrases:
                phrases.append(phrase)
            if len(phrases) >= 4:
                break

        activity_word = "activity" if patched_count == 1 else "activities"
        if not phrases:
            return f"Applied code-level fixes across {patched_count} {activity_word}."

        if len(phrases) == 1:
            joined = phrases[0]
        elif len(phrases) == 2:
            joined = f"{phrases[0]} and {phrases[1]}"
        else:
            joined = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
        return f"{joined.capitalize()} across {patched_count} {activity_word}."

    # ---- code patch generation --------------------------------------

    def _findings_for_activity(self, state: InvestigationState, activity_name: str) -> list[dict]:
        return [
            f.to_dict() for f in state.validated_findings
            if activity_name in (f.affected_activities or [])
        ]

    def _recommendations_for_activity(self, state: InvestigationState, activity_name: str) -> list[dict]:
        # target_activity is a free-text label (e.g. "Bronze SynapseNotebook -
        # df.count() Call") that CONTAINS the real activity_name rather than
        # equalling it, so match by substring, not equality.
        out = []
        for r in state.recommendations:
            if r.target_activity and activity_name in r.target_activity:
                out.append({
                    "title": r.title,
                    "description": r.description,
                    "impact_score": r.impact_score,
                    "effort_score": r.effort_score,
                    "risk_score": r.risk_score,
                    "confidence": r.confidence,
                })
        return out

    def _no_issue_result(self, reason: str) -> dict:
        return {"has_issues": False, "patched_code": None, "explanation": reason}

    async def _call_code_patch_model(self, prompt: str) -> str:
        if anthropic_ready():
            client = anthropic.AsyncAnthropic(api_key=get_secret("ANTHROPIC_API_KEY", required=True))
            response = await client.messages.create(
                model=CODE_PATCH_MODEL,
                max_tokens=8000,
                system=CODE_PATCH_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in response.content if hasattr(b, "text"))
        if SDK_AVAILABLE:
            options = ClaudeAgentOptions(model=CODE_PATCH_MODEL, system_prompt=CODE_PATCH_SYSTEM)
            raw_text = ""
            async for message in sdk_query(prompt=prompt, options=options):
                text_piece = getattr(message, "text", None)
                if text_piece:
                    raw_text += text_piece
            return raw_text
        raise RuntimeError(
            f"{self.role_name}: no LLM provider available for code patch generation "
            "(set ANTHROPIC_API_KEY, or install claude_agent_sdk)."
        )

    @staticmethod
    def _strip_fence(text: str) -> str:
        t = (text or "").strip()
        if t.startswith("```"):
            t = t.split("```", 2)[1] if t.count("```") >= 2 else t.lstrip("`")
            t = t[4:] if t.lower().startswith("json") else t
            t = t.strip()
        if not t.startswith("{"):
            start, end = t.find("{"), t.rfind("}")
            if start != -1 and end != -1:
                t = t[start:end + 1]
        return t

    async def generate_code_patches(self, state: InvestigationState) -> dict:
        """For every enriched notebook activity: if evidence-backed findings
        or recommendations target it, ask the model for the full corrected
        source + an explanation. If nothing targets it, skip the LLM call
        entirely and record that no code-level issue was found."""
        ctx = state.context
        notebooks = (getattr(ctx, "enrichment", None) or {}).get("notebooks", {}) or {}
        patches: dict[str, dict] = {}

        for activity_name, nb in notebooks.items():
            source = nb.get("source_code") or ""
            findings = self._findings_for_activity(state, activity_name)
            recs = self._recommendations_for_activity(state, activity_name)

            if not findings and not recs:
                patches[activity_name] = self._no_issue_result(
                    f"No verified findings or recommendations target '{activity_name}'. "
                    "No code-level changes suggested for this notebook."
                )
                continue

            payload = {
                "activity_name": activity_name,
                "language": nb.get("language"),
                "current_source_code": source,
                "verified_findings": findings,
                "recommendations": recs,
            }
            prompt = (
                "Fix the following notebook based ONLY on the findings/"
                "recommendations provided:\n"
                f"{json.dumps(payload, indent=2, default=str)}\n"
                "Return the JSON object described in your system prompt."
            )
            try:
                raw = await self._call_code_patch_model(prompt)
                data = json.loads(self._strip_fence(raw))
                has_issues = bool(data.get("has_issues"))
                patches[activity_name] = {
                    "has_issues": has_issues,
                    "patched_code": data.get("patched_code") if has_issues else None,
                    "explanation": data.get("explanation") or "",
                }
            except Exception as e:
                # Fail loudly-but-gracefully into the report rather than
                # crashing report generation - the raw findings are still
                # visible elsewhere in the investigation output.
                patches[activity_name] = self._no_issue_result(
                    f"Automatic patch generation failed for '{activity_name}' "
                    f"({type(e).__name__}: {e}). See findings/recommendations "
                    "above for manual remediation."
                )
        return patches

    async def generate_final_report(self, state: InvestigationState) -> str:
        ctx = state.context
        self._last_call_meta = {}
        self._last_provider = "deterministic"
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
                self._last_provider = "azure_openai"
                self._last_call_meta = dict(azure_openai_client.last_call_meta)
            except Exception as e:
                self._last_provider = "deterministic_fallback"
                self._last_call_meta = {"error": f"{type(e).__name__}: {e}"}
        return narrative
    
    def build_cost_summary(self, state: InvestigationState) -> dict | None:
        cost_agent_ran = any(
            f.agent == "cost_intelligence_agent"
            for f in state.validated_findings
        )
        if not cost_agent_ran:
            return None

        payload = state.context.raw_payload or {}
        current_cost = float(payload.get("total_cost", 0))
        baseline = float(payload.get("baseline_monthly_cost", 0))
        currency = payload.get("currency", "INR")

        today = datetime.now(timezone.utc)
        days_elapsed = max(today.day, 1)
        days_in_month = monthrange(today.year, today.month)[1]

        estimated_monthly_cost = round(
            (current_cost / days_elapsed) * days_in_month, 2
        )
        forecast_deviation = round(estimated_monthly_cost - baseline, 2)
        forecast_deviation_pct = (
            round((forecast_deviation / baseline) * 100, 2) if baseline else 0
        )

        if forecast_deviation > 0:
            status = "above_baseline"
        elif forecast_deviation < 0:
            status = "below_baseline"
        else:
            status = "within_baseline"

        return {
            "currency": currency,
            "current_month_cost": round(current_cost, 2),
            "baseline_monthly_cost": round(baseline, 2),
            "estimated_monthly_cost": estimated_monthly_cost,
            "forecast_deviation": forecast_deviation,
            "forecast_deviation_pct": forecast_deviation_pct,
            "forecast_status": status,
        }

    async def generate(self, state: InvestigationState) -> dict:
        narrative = await self.generate_final_report(state)
        roadmap = self.create_action_roadmap(state)
        fixes = self.generate_suggested_fixes(state)
        code_patches = await self.generate_code_patches(state)
        compilation_adjustment = self.generate_compilation_adjustment(code_patches)
        ctx = state.context
        apply_fix_payloads = [self.prepare_apply_fix_payload(f, ctx) for f in fixes]
        cost_summary = self.build_cost_summary(state)
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "subscription_id": ctx.subscription_id,
            "service": ctx.service,
            "resource_group": ctx.resource_group,
            "workspace_name": ctx.workspace_name,
            "item_name": ctx.item_name,
            "executive_summary": narrative,
            "compilation_adjustment": compilation_adjustment,
            "pipeline_health": ctx.pipeline_health,
            "pipeline_deviation_pct": ctx.pipeline_deviation_pct,
            "root_causes": [
                {"category": rc.category, "description": rc.description, "confidence": rc.confidence}
                for rc in state.root_causes
            ],
            "action_roadmap": roadmap,
            "suggested_fixes": fixes,
            "apply_fix_payloads": apply_fix_payloads,
            "code_patches": code_patches,
        }
        if cost_summary is not None:
            report.update(cost_summary)

        return report