"""
Sentiment Agent - the only agent that reads free text.

Design decisions worth defending:

1. ONE CALL PER CUSTOMER, not per conversation. A customer with 5 chats
   gets all 5 in a single prompt, keyed by conversation_id. This cuts the
   run from 487 calls to ~169, which matters on a free tier, and it lets
   the model see the customer's trajectory rather than isolated snapshots.

2. CLOSED VOCABULARY. Intents, sentiments and severities come from fixed
   lists. Free-form labels cannot be aggregated, weighted, or filtered.

3. MANDATORY VERBATIM EVIDENCE. Every flag must be justified by an exact
   quote from the transcript. This is the anti-hallucination mechanism:
   the model is never asked for facts about the customer, only to interpret
   text we supplied, and it must point at the words it used.

4. RECENCY WEIGHTING. A furious message from 3 days ago matters more than
   one from 80 days ago. Weight halves every 30 days.
"""

import math

from pydantic import ValidationError

from src.agents.base_agent import BaseAgent
from src.llm_client import LLMClient, LLMUnavailable
from src.llm_schemas import BatchAnalysis
from src.models import AgentResult, Signal

SOURCE = "conversation"
RECENCY_HALF_LIFE_DAYS = 30
FRUSTRATION_HIGH = 7


PROMPT = """You are a customer-retention analyst reviewing support conversations.

Analyse EACH conversation below and return structured findings.

RULES
- Return ONLY valid JSON. No markdown, no commentary, no code fences.
- Use ONLY the allowed values listed for each field.
- key_evidence must contain EXACT VERBATIM quotes copied from the transcript.
  Never paraphrase. Never invent. If there is no supporting quote for a flag,
  set that flag to false.
- Judge only what is written. Do not infer account history you cannot see.
- A customer being angry is NOT the same as a customer leaving. Set
  churn_intent true only for explicit exit language (cancelling, offboarding,
  not renewing, switching vendor).

ALLOWED VALUES
sentiment:       positive | neutral | negative | very_negative
primary_intent:  billing_dispute | technical_issue | cancellation_intent |
                 feature_request | general_query | complaint | praise |
                 onboarding_help
urgency:         low | medium | high

FIELD DEFINITIONS
frustration_level    0-10 integer. 0 = content, 10 = furious.
churn_intent         true if they state or strongly imply they are leaving.
competitor_mentioned true if another vendor or alternative is referenced.
escalation_language  true if they demand a manager, threaten legal action,
                     or mention going public.
repeat_issue         true if they say this has happened before or that they
                     have contacted support about it previously.
summary              one sentence, under 20 words, written for a retention
                     agent about to phone this customer.

OUTPUT SCHEMA
{{
  "conversations": [
    {{
      "conversation_id": "<echo the id exactly>",
      "sentiment": "...",
      "frustration_level": 0,
      "primary_intent": "...",
      "urgency": "...",
      "churn_intent": false,
      "competitor_mentioned": false,
      "escalation_language": false,
      "repeat_issue": false,
      "key_evidence": ["exact quote", "exact quote"],
      "summary": "..."
    }}
  ]
}}

EXAMPLE
Input:
[CONV99999] Customer: This is the third time the export has failed.
Agent: I'm sorry, escalating now.
Customer: I've already got a trial running with another provider.

Output:
{{"conversations":[{{"conversation_id":"CONV99999","sentiment":"very_negative",
"frustration_level":8,"primary_intent":"complaint","urgency":"high",
"churn_intent":false,"competitor_mentioned":true,"escalation_language":false,
"repeat_issue":true,
"key_evidence":["This is the third time the export has failed",
"I've already got a trial running with another provider"],
"summary":"Repeat export failures; customer is trialling an alternative provider."}}]}}

CONVERSATIONS TO ANALYSE
{conversations}
"""


def recency_weight(days_ago):
    """Exponential decay: weight halves every 30 days.

    A 3-day-old complaint scores ~0.93; an 80-day-old one ~0.16. Chosen for
    explainability - 'importance halves every month' is a sentence an ops
    manager can understand and challenge.
    """
    if days_ago is None or days_ago < 0:
        return 1.0
    return round(0.5 ** (days_ago / RECENCY_HALF_LIFE_DAYS), 3)


def _format_conversations(conversations):
    blocks = []
    for c in conversations:
        blocks.append(f"[{c['conversation_id']}] ({c.get('channel', 'chat')}, "
                      f"{c.get('days_ago', 0)} days ago)\n{c['transcript']}")
    return "\n\n".join(blocks)


# Keywords used to match a quote to the signal it actually supports.
# Round-robin assignment produced evidence that contradicted its own label -
# a COMPETITOR_MENTION justified by a quote about ticket counts. Evidence
# that does not support its claim is worse than no evidence, because it
# looks like proof while providing none.
EVIDENCE_KEYWORDS = {
    "CHURN_INTENT": ("cancel", "cancelling", "terminate", "not renew",
                     "non-renew", "offboard", "leaving", "discontinue",
                     "close our account", "end of the term", "contract to end",
                     "signed with", "moving to", "switching to", "wind down"),
    "COMPETITOR_MENTION": ("competitor", "alternative", "another provider",
                           "another vendor", "another platform", "other vendor",
                           "rival", "elsewhere", "trialling", "trialing",
                           "signed with", "moving to", "switching to"),
    "ESCALATION_LANGUAGE": ("manager", "supervisor", "escalat", "legal",
                            "ombudsman", "twitter", "social media", "director",
                            "formal complaint", "done chasing"),
    "REPEAT_ISSUE": ("third time", "second time", "again", "keeps", "still no",
                     "every week", "repeatedly", "same issue", "multiple times",
                     "already contacted", "chasing", "none properly resolved",
                     "four tickets", "two months"),
}


def pick_evidence(analysis, signal_name, used):
    """Choose the quote that best supports THIS signal.

    Scores each quote by how many of the signal's keywords it contains, and
    prefers quotes not already used by another signal so the drill-down shows
    different evidence for different findings. Falls back to the summary when
    no quote is available.
    """
    quotes = [q for q in analysis.key_evidence if q]
    if not quotes:
        return analysis.summary

    keywords = EVIDENCE_KEYWORDS.get(signal_name, ())

    def score(quote):
        text = quote.lower()
        hits = sum(1 for k in keywords if k in text)
        # Small penalty for reuse: relevance still wins over novelty.
        return hits * 10 - (3 if quote in used else 0)

    best = max(quotes, key=score)
    # If nothing matched at all, fall back to any unused quote.
    if not any(k in best.lower() for k in keywords):
        unused = [q for q in quotes if q not in used]
        best = unused[0] if unused else quotes[0]
    return best


def signals_from_analysis(analysis, days_ago):
    """Map one validated ConversationAnalysis onto zero or more Signals.

    Kept as a pure function so it can be unit-tested without an API key.
    """
    w = recency_weight(days_ago)
    out = []
    used = set()

    def quote(signal_name):
        q = pick_evidence(analysis, signal_name, used)
        used.add(q)
        return q

    def add(name, severity, evidence, value=None):
        out.append(Signal(name=name, severity=severity, source=SOURCE,
                          evidence=evidence, value=value, recency_weight=w))

    if analysis.churn_intent:
        add("CHURN_INTENT", "critical",
            f'Stated intent to leave: "{quote("CHURN_INTENT")}"')

    if analysis.competitor_mentioned:
        add("COMPETITOR_MENTION", "high",
            f'Referenced an alternative provider: "{quote("COMPETITOR_MENTION")}"')

    if analysis.escalation_language:
        add("ESCALATION_LANGUAGE", "high",
            f'Escalation demanded: "{quote("ESCALATION_LANGUAGE")}"')

    if analysis.repeat_issue:
        add("REPEAT_ISSUE", "high",
            f'Recurring unresolved problem: "{quote("REPEAT_ISSUE")}"')

    if analysis.frustration_level >= FRUSTRATION_HIGH:
        severity = "high" if analysis.frustration_level >= 9 else "medium"
        add("HIGH_FRUSTRATION", severity,
            f"Frustration {analysis.frustration_level}/10 - {analysis.summary}",
            value=analysis.frustration_level)

    return out


class SentimentAgent(BaseAgent):
    name = "sentiment"

    def __init__(self, client=None):
        self.client = client or LLMClient()

    def analyse(self, context):
        conversations = context.get("conversations", [])

        # Silence is a signal. A customer with no contact is not neutral -
        # they may be disengaging quietly, which is the hardest case to catch.
        if not conversations:
            return AgentResult(
                agent_name=self.name,
                signals=[Signal(
                    name="NO_CONTACT", severity="low", source=SOURCE,
                    evidence="No support conversations on record in 90 days",
                )],
                confidence=1.0,
            )

        prompt = PROMPT.format(conversations=_format_conversations(conversations))

        try:
            raw = self.client.generate_json(prompt)
        except LLMUnavailable as exc:
            # Degrade, don't die. The rule agents still produce a score;
            # we just mark the assessment as less confident.
            return AgentResult(agent_name=self.name, signals=[],
                               confidence=0.0, error=str(exc))

        try:
            batch = BatchAnalysis.model_validate(raw)
        except ValidationError as exc:
            return AgentResult(
                agent_name=self.name,
                signals=[Signal(
                    name="PARSE_FAILED", severity="low", source=SOURCE,
                    evidence="Conversation analysis could not be validated; "
                             "scored on behavioural data only",
                )],
                confidence=0.0,
                error=f"ValidationError: {exc.error_count()} field(s) invalid",
            )

        days_by_id = {c["conversation_id"]: c.get("days_ago", 0)
                      for c in conversations}

        signals = []
        for analysis in batch.conversations:
            signals.extend(signals_from_analysis(
                analysis, days_by_id.get(analysis.conversation_id, 0)))

        signals = self._dedupe(signals)
        return AgentResult(agent_name=self.name, signals=signals,
                           confidence=1.0)

    @staticmethod
    def _dedupe(signals):
        """A customer who mentions a competitor in three conversations should
        produce ONE competitor signal, not three. Keep the highest-severity
        instance, and among equals the most recent."""
        best = {}
        for s in signals:
            current = best.get(s.name)
            if (current is None
                    or s.severity_rank > current.severity_rank
                    or (s.severity_rank == current.severity_rank
                        and s.recency_weight > current.recency_weight)):
                best[s.name] = s
        return list(best.values())