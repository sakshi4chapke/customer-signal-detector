"""
Recommendation Agent - turns a score into something a human can act on.

A score tells operations WHO to call. This tells them WHAT to do and WHAT
to say. That difference is what separates a dashboard from a decision tool.

Two design decisions worth defending:

1. CLOSED ACTION VOCABULARY. The LLM chooses from eight fixed actions. Free
   text like "reach out warmly and rebuild trust" cannot be routed to a
   team, counted in a weekly report, or measured for effectiveness. A fixed
   action can be all three.

2. ROUTING METADATA IS CONFIG, NOT LLM OUTPUT. Owner team, SLA and priority
   are looked up deterministically from the chosen action and the risk band.
   Asking the model to invent an SLA would produce plausible-sounding numbers
   with no organisational meaning.
"""

from pydantic import ValidationError

from src.agents.base_agent import BaseAgent
from src.llm_client import LLMClient, LLMUnavailable
from src.llm_schemas import Recommendation
from src.models import AgentResult
from src.config import SIGNAL_WEIGHTS, SEVERITY_MULTIPLIER

# ---------------------------------------------------------------- routing --
# Which team owns each action, and how quickly they must act. Config, so an
# operations lead can change the SLA without touching code.
ACTION_ROUTING = {
    "IMMEDIATE_CALL":       {"team": "Senior Retention", "sla_hours": 24},
    "RETENTION_OFFER":      {"team": "Retention (approval required)", "sla_hours": 48},
    "BILLING_REVIEW":       {"team": "Finance",          "sla_hours": 48},
    "TECHNICAL_ESCALATION": {"team": "Engineering",      "sla_hours": 24},
    "PROACTIVE_CHECKIN":    {"team": "Customer Success", "sla_hours": 72},
    "ONBOARDING_SUPPORT":   {"team": "Onboarding",       "sla_hours": 72},
    "USAGE_NUDGE":          {"team": "Lifecycle Marketing", "sla_hours": 168},
    "MONITOR_ONLY":         {"team": "None",             "sla_hours": None},
}

PRIORITY_BY_LEVEL = {"Critical": "P1", "High": "P2",
                     "Medium": "P3", "Low": "P4"}


PROMPT = """You are advising a customer-retention agent who has 30 seconds
before phoning this customer. Write for them, not for an analyst.

TASK
Choose ONE action and write the briefing.

ALLOWED ACTIONS - choose exactly one
IMMEDIATE_CALL        explicit exit intent or competitor engagement; needs a
                      senior agent on the phone today
RETENTION_OFFER       value or pricing is the core objection; a discount or
                      credit could plausibly save the account
BILLING_REVIEW        failed payments, disputed charges, or billing confusion
TECHNICAL_ESCALATION  unresolved technical faults are the root cause
PROACTIVE_CHECKIN     declining engagement with no complaint; a human should
                      make contact before it worsens
ONBOARDING_SUPPORT    new customer who never got started properly
USAGE_NUDGE           mild disengagement; automated re-engagement is enough
MONITOR_ONLY          no meaningful risk; review next cycle

RULES
- Return ONLY valid JSON. No markdown, no commentary.
- Reference ONLY the signals listed below. Invent nothing.
- Quote the customer's own words where a quote is provided.
- Write in plain English. Do not mention scores, models, algorithms, or
  "our system". The agent needs facts about the customer, not about the tool.
- Do not use the word "churn". Say what will actually happen.
- talking_points must be things the agent can literally say or ask on the
  call, not internal observations.

OUTPUT SCHEMA
{{
  "action": "ONE_OF_THE_ALLOWED_ACTIONS",
  "explanation": "2-3 sentences: what is happening and why it matters now",
  "talking_points": ["...", "...", "..."]
}}

EXAMPLE OUTPUT
{{"action":"IMMEDIATE_CALL","explanation":"This customer has told support they plan to cancel at the end of their term and is already trialling another provider. Four tickets in two months went unresolved, and their usage has fallen by three quarters. The contract ends in 21 days, so there is a narrow window to intervene.","talking_points":["Acknowledge the four unresolved tickets directly and apologise for the pattern, not just the individual faults","Ask what the alternative platform does better - the answer determines whether this is recoverable","Offer a named technical owner for the outstanding issues before discussing commercial terms"]}}
"""


def _format_profile(customer, assessment):
    return (
        f"Segment: {customer['segment']} · Plan: {customer['plan_tier']} · "
        f"Monthly revenue: {customer['monthly_revenue']:,.0f}\n"
        f"Tenure: {int(customer['tenure_months'])} months · "
        f"Contract renews in {int(customer['contract_end_days'])} days\n"
        f"Risk band: {assessment['risk_level']}"
    )


def _rank_by_impact(signals):
    """Order signals by how much they actually contributed to the score.

    Sorting by severity alone is misleading: several signals share the
    "critical" label, so the briefing could lead with "11 support tickets"
    while omitting that the customer said they are cancelling. Weight x
    severity x recency is what the score used, so it is what the briefing
    should lead with too.
    """
    def impact(s):
        return (SIGNAL_WEIGHTS.get(s["name"], 0)
                * SEVERITY_MULTIPLIER.get(s["severity"], 1.0)
                * s.get("recency_weight", 1.0))
    return sorted(signals, key=impact, reverse=True)


def _format_signals(signals):
    return "\n".join(f"- [{s['severity']}] {s['name']}: {s['evidence']}"
                     for s in _rank_by_impact(signals)[:10])


# ------------------------------------------------------- rules fallback ----

def fallback_recommendation(signals, risk_level):
    """Deterministic recommendation when the LLM is unavailable.

    Less eloquent than the model, but never absent. The pipeline must still
    produce an actionable output when quota is exhausted - that is the whole
    point of designing for degradation rather than assuming availability.
    """
    names = {s["name"] for s in signals}

    if "CHURN_INTENT" in names or "COMPETITOR_MENTION" in names:
        action = "IMMEDIATE_CALL"
    elif "PAYMENT_FAILURE" in names or "OVERDUE" in names:
        action = "BILLING_REVIEW"
    elif "UNRESOLVED_BACKLOG" in names or "SLOW_RESOLUTION" in names:
        action = "TECHNICAL_ESCALATION"
    elif "DOWNGRADE" in names or "LOW_NPS" in names:
        action = "RETENTION_OFFER"
    elif "NEW_CUSTOMER_FRAGILE" in names:
        action = "ONBOARDING_SUPPORT"
    elif "INACTIVITY" in names or "USAGE_DECLINE" in names:
        action = ("PROACTIVE_CHECKIN" if risk_level in ("High", "Critical")
                  else "USAGE_NUDGE")
    elif names:
        action = "PROACTIVE_CHECKIN" if risk_level != "Low" else "MONITOR_ONLY"
    else:
        action = "MONITOR_ONLY"

    top = _rank_by_impact(signals)[:3]
    explanation = (("Flagged on: " + "; ".join(s["evidence"] for s in top) + ".")
                   if top else "No significant risk signals detected.")

    return Recommendation(
        action=action,
        explanation=explanation[:600],
        talking_points=([s["evidence"][:180] for s in top]
                        or ["No specific concerns to raise at this time."]),
    )


class RecommendationAgent(BaseAgent):
    name = "recommendation"

    def __init__(self, client=None, use_llm=True):
        self.use_llm = use_llm
        self.client = client if client is not None else (
            LLMClient() if use_llm else None)

    def analyse(self, context):
        customer = context["customer"]
        assessment = context["assessment"]
        signals = context["signals"]          # list of dicts

        rec, source, error = None, "llm", None

        if self.client is not None and signals:
            prompt = (PROMPT
                      + "\n\nCUSTOMER\n" + _format_profile(customer, assessment)
                      + "\n\nDETECTED SIGNALS\n" + _format_signals(signals))
            try:
                raw = self.client.generate_json(prompt)
                rec = Recommendation.model_validate(raw)
            except (LLMUnavailable, ValidationError) as exc:
                error = f"{type(exc).__name__}: {str(exc)[:120]}"

        if rec is None:
            rec = fallback_recommendation(signals, assessment["risk_level"])
            source = "rules"

        routing = ACTION_ROUTING[rec.action]
        result = AgentResult(agent_name=self.name, signals=[],
                             confidence=1.0 if source == "llm" else 0.6,
                             error=error)
        result.recommendation = {
            "action": rec.action,
            "priority": PRIORITY_BY_LEVEL[assessment["risk_level"]],
            "owner_team": routing["team"],
            "sla_hours": routing["sla_hours"],
            "explanation": rec.explanation,
            "talking_points": rec.talking_points,
            "generated_by": source,
        }
        return result