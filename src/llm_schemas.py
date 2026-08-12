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