"""
Signal Engine - merges signals from every agent into one list per customer.

Three jobs:
  1. LEFT JOIN semantics. Customers with no conversations must survive.
     They are disproportionately quiet drifters - the highest-risk group -
     so an inner join would silently delete the customers we most want.
  2. Deduplication. If two agents report the same fact, count it once.
  3. Uniform shape. The scorer receives one flat list per customer and does
     not need to know which agent produced what.
"""

import json
import os

import pandas as pd

from src.agents.behaviour_agent import BehaviourAgent
from src.agents.billing_agent import BillingAgent
from src.models import Signal

PROC_DIR = os.path.join("data", "processed")


def load_conversation_signals(path=None):
    """Read the cached Sentiment Agent output. Returns {} if absent, so the
    pipeline still runs rules-only without the LLM stage."""
    path = path or os.path.join(PROC_DIR, "conversation_signals.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _signals_from_dicts(records):
    out = []
    for d in records:
        out.append(Signal(
            name=d["name"], severity=d["severity"], source=d["source"],
            evidence=d["evidence"], value=d.get("value"),
            recency_weight=d.get("recency_weight", 1.0),
        ))
    return out


def dedupe(signals):
    """One signal per name. Keep the highest severity; among equals, the most
    recent. A customer who mentions a competitor three times has ONE
    competitor problem, not three."""
    best = {}
    for s in signals:
        cur = best.get(s.name)
        if (cur is None
                or s.severity_rank > cur.severity_rank
                or (s.severity_rank == cur.severity_rank
                    and s.recency_weight > cur.recency_weight)):
            best[s.name] = s
    # Sort by severity descending so the UI shows the worst first.
    return sorted(best.values(),
                  key=lambda s: (-s.severity_rank, -s.recency_weight))


def build_customer_signals(customers_df, conversation_signals=None):
    """Run the rule agents and merge in cached conversation signals.

    Returns {customer_id: {"signals": [...], "confidence": float,
                           "agent_trace": {...}}}
    """
    conversation_signals = (conversation_signals
                            if conversation_signals is not None
                            else load_conversation_signals())

    behaviour, billing = BehaviourAgent(), BillingAgent()
    out = {}

    for row in customers_df.to_dict("records"):
        cid = row["customer_id"]
        context = {"customer": row}

        rb = behaviour.run(context)
        rl = billing.run(context)

        conv = conversation_signals.get(cid, {})
        conv_signals = _signals_from_dicts(conv.get("signals", []))

        # Confidence reflects how much of the picture we actually saw.
        # Rules always run, so we are never at zero - but an assessment made
        # without the conversation layer is genuinely less certain, and the
        # UI should say so rather than pretending otherwise.
        if not conversation_signals:
            confidence = 0.5                      # LLM stage never ran
        elif conv.get("error"):
            confidence = 0.4                      # LLM failed for this customer
        elif conv.get("skipped"):
            confidence = 0.6                      # outside the LLM budget
        else:
            confidence = 1.0
        if not rb.ok or not rl.ok:
            confidence = min(confidence, 0.5)

        merged = dedupe(rb.signals + rl.signals + conv_signals)

        out[cid] = {
            "signals": merged,
            "confidence": round(confidence, 2),
            "agent_trace": {
                "behaviour": {"count": len(rb.signals), "error": rb.error,
                              "latency_ms": rb.latency_ms},
                "billing": {"count": len(rl.signals), "error": rl.error,
                            "latency_ms": rl.latency_ms},
                "sentiment": {"count": len(conv_signals),
                              "error": conv.get("error"),
                              "skipped": conv.get("skipped")},
            },
        }

    return out


def signals_to_frame(customer_signals):
    """Flatten to a long dataframe - one row per (customer, signal).
    Useful for the heatmap and signal-frequency charts."""
    rows = []
    for cid, data in customer_signals.items():
        for s in data["signals"]:
            rows.append({"customer_id": cid, **s.to_dict()})
    return pd.DataFrame(rows)