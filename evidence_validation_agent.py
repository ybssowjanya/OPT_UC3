"""
Evidence Validation Agent
Model: Claude Sonnet 4

Tools: cross_validate_findings, score_evidence_confidence,
       deduplicate_and_merge_findings, validate_against_original_deviation,
       generate_validated_evidence_package
"""
from __future__ import annotations
from schemas import InvestigationContext, AgentFinding

import os

MIN_CONFIDENCE_TO_VERIFY = float(os.environ.get("MIN_CONFIDENCE_TO_VERIFY", "0.55"))


class EvidenceValidationAgent:
    role_name = "evidence_validation_agent"
    model = "claude-sonnet-4-6"

    def cross_validate_findings(self, findings: list[AgentFinding]) -> dict[str, list[AgentFinding]]:
        by_activity: dict[str, list[AgentFinding]] = {}
        for f in findings:
            for act in (f.affected_activities or ["__pipeline_level__"]):
                by_activity.setdefault(act, []).append(f)
        return by_activity

    def deduplicate_and_merge_findings(self, findings: list[AgentFinding]) -> list[AgentFinding]:
        seen: dict[tuple, AgentFinding] = {}
        for f in findings:
            key = (tuple(sorted(f.affected_activities)), f.summary[:80])
            if key not in seen:
                seen[key] = f
            else:
                # Merge: keep higher confidence, union evidence
                existing = seen[key]
                if f.confidence > existing.confidence:
                    existing.confidence = f.confidence
                existing.evidence.update(f.evidence)
        return list(seen.values())

    def validate_against_original_deviation(self, ctx: InvestigationContext, finding: AgentFinding) -> bool:
        known_activity_names = {a.activity_name for a in ctx.activities}
        degraded_names = {a.activity_name for a in ctx.degraded_activities()}
        if not finding.affected_activities:
            return True  # pipeline-level finding, nothing to check
       
        refs_known = any(n in known_activity_names for n in finding.affected_activities)
        refs_degraded = any(n in degraded_names for n in finding.affected_activities)
        return refs_known and (refs_degraded or not degraded_names)

    def score_evidence_confidence(self, ctx: InvestigationContext, finding: AgentFinding,
                                   corroborating_count: int) -> float:
        base = finding.confidence
        # Multiple independent agents flagging the same activity raises confidence.
        boost = min(0.15 * (corroborating_count - 1), 0.3) if corroborating_count > 1 else 0.0
        return min(1.0, base + boost)

    async def validate(self, ctx: InvestigationContext, findings: list[AgentFinding]) -> list[AgentFinding]:
        agents_by_activity: dict[str, set] = {}
        for f in findings:
            for act in (f.affected_activities or ["__pipeline_level__"]):
                agents_by_activity.setdefault(act, set()).add(f.agent)

        merged = self.deduplicate_and_merge_findings(findings)

        validated: list[AgentFinding] = []
        for f in merged:
            if not self.validate_against_original_deviation(ctx, f):
                f.status = "rejected"
                continue

            corroborating = max(
                (len(agents_by_activity.get(act, set()))
                 for act in (f.affected_activities or ["__pipeline_level__"])),
                default=1,
            )
            f.confidence = self.score_evidence_confidence(ctx, f, corroborating)

            if f.confidence >= MIN_CONFIDENCE_TO_VERIFY:
                f.status = "verified"
                validated.append(f)
            else:
                f.status = "unverified"

        return self.generate_validated_evidence_package(validated)

    def generate_validated_evidence_package(self, validated: list[AgentFinding]) -> list[AgentFinding]:
        return sorted(validated, key=lambda f: f.confidence, reverse=True)

