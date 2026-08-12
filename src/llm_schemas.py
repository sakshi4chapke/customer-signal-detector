"""
Schemas for LLM output.

This is the trust boundary. Everything above this line in the pipeline is
constructed by our own code and uses dataclasses; everything arriving from
Gemini passes through pydantic first.

Why it matters: an LLM can return frustration_level of 47, an invented
intent category, or a string where a boolean should be. Without validation
those values flow straight into the risk score and quietly corrupt it.
Pydantic converts "wrong data" into "a caught exception", which the agent
can retry or degrade from.
"""

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator

SENTIMENTS = ("positive", "neutral", "negative", "very_negative")
INTENTS = (
    "billing_dispute", "technical_issue", "cancellation_intent",
    "feature_request", "general_query", "complaint", "praise",
    "onboarding_help",
)
URGENCIES = ("low", "medium", "high")


class ConversationAnalysis(BaseModel):
    """One analysed conversation. Field names mirror the prompt exactly."""

    conversation_id: str

    sentiment: Literal[SENTIMENTS]
    frustration_level: int = Field(ge=0, le=10)
    primary_intent: Literal[INTENTS]
    urgency: Literal[URGENCIES]

    churn_intent: bool
    competitor_mentioned: bool
    escalation_language: bool
    repeat_issue: bool

    key_evidence: List[str] = Field(default_factory=list, max_length=4)
    summary: str = Field(max_length=200)

    @field_validator("key_evidence")
    @classmethod
    def evidence_must_be_substantive(cls, v):
        """Drop empty or single-word 'quotes'. An evidence field containing
        'yes' is worse than no evidence at all - it looks like support for a
        claim while providing none."""
        return [q.strip() for q in v if q and len(q.strip()) > 8]

    @field_validator("summary")
    @classmethod
    def summary_not_empty(cls, v):
        if not v.strip():
            raise ValueError("summary cannot be empty")
        return v.strip()


class BatchAnalysis(BaseModel):
    """What the model returns for one customer: all their conversations."""
    conversations: List[ConversationAnalysis]


# --------------------------------------------------------------------------
# RECOMMENDATION (Phase 7)
# --------------------------------------------------------------------------
# The action vocabulary is CLOSED. Free-form actions cannot be routed to a
# team, counted in a report, or measured for effectiveness. "Reach out warmly
# and rebuild trust" is not a work item; RETENTION_OFFER is.
ACTIONS = (
    "IMMEDIATE_CALL",         # senior agent phones within 24h
    "RETENTION_OFFER",        # discount or credit, needs approval
    "BILLING_REVIEW",         # finance investigates charges
    "TECHNICAL_ESCALATION",   # engineering ticket, priority raised
    "PROACTIVE_CHECKIN",      # CSM email within 3 days
    "ONBOARDING_SUPPORT",     # re-run onboarding or training session
    "USAGE_NUDGE",            # automated re-engagement campaign
    "MONITOR_ONLY",           # no action, review next cycle
)


class Recommendation(BaseModel):
    """What the Recommendation Agent returns for one customer."""

    action: Literal[ACTIONS]
    explanation: str = Field(min_length=20, max_length=600)
    talking_points: List[str] = Field(min_length=1, max_length=4)

    @field_validator("explanation")
    @classmethod
    def no_jargon(cls, v):
        """The explanation is read by a retention agent with 30 seconds
        before dialling, not by a data scientist."""
        banned = ("churn score", "risk score of", "the model", "algorithm",
                  "weighted sum", "our system predicts")
        low = v.lower()
        for term in banned:
            if term in low:
                raise ValueError(f"explanation should avoid jargon: '{term}'")
        return v.strip()

    @field_validator("talking_points")
    @classmethod
    def points_substantive(cls, v):
        cleaned = [p.strip() for p in v if p and len(p.strip()) > 10]
        if not cleaned:
            raise ValueError("at least one usable talking point required")
        return cleaned