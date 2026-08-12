"""
Risk Assessment Agent - turns a list of signals into one defensible number.

Deliberately deterministic. The LLM is never asked to score a customer,
because:

  * a model's numeric judgement drifts between runs, so the same customer
    would rank differently each morning;
  * an ops manager cannot challenge a number the system cannot explain;
  * an auditor cannot reproduce it.

Machines do arithmetic, LLMs do language. This agent is the arithmetic.
"""

from src.agents.base_agent import BaseAgent
from src.models import AgentResult
from src.config import (
    SIGNAL_WEIGHTS, SEVERITY_MULTIPLIER, saturate, risk_level,
)


class RiskAgent(BaseAgent):
    name = "risk"

    def score(self, signals):
        """Weighted sum, then exponential saturation.

            contribution = weight x severity_multiplier x recency_weight

        Recency only varies for conversation signals; structured data is a
        current snapshot and carries weight 1.0.
        """
        breakdown = []
        raw = 0.0

        for s in signals:
            weight = SIGNAL_WEIGHTS.get(s.name, 0)
            if weight == 0:
                continue
            mult = SEVERITY_MULTIPLIER[s.severity]
            contribution = weight * mult * s.recency_weight
            raw += contribution
            breakdown.append({
                "signal": s.name,
                "severity": s.severity,
                "source": s.source,
                "weight": weight,
                "severity_multiplier": mult,
                "recency_weight": s.recency_weight,
                "contribution": round(contribution, 2),
                "evidence": s.evidence,
            })

        breakdown.sort(key=lambda b: -b["contribution"])
        final = saturate(raw)
        return final, round(raw, 2), breakdown

    def analyse(self, context):
        signals = context["signals"]
        final, raw, breakdown = self.score(signals)

        result = AgentResult(agent_name=self.name, signals=[], confidence=1.0)
        # The scorer produces an assessment rather than signals, so it is
        # attached to the result for the orchestrator to pick up.
        result.assessment = {
            "risk_score": final,
            "raw_score": raw,
            "risk_level": risk_level(final),
            "score_breakdown": breakdown,
            "top_contributors": [b["signal"] for b in breakdown[:3]],
        }
        return result