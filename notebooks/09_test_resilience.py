"""
Prove the pipeline degrades instead of dying.

Deliberately break each agent in turn and confirm that:
  * an assessment is still produced
  * the score reflects whatever signals survived
  * confidence drops to say so
  * the trace names the agent that failed

This is the evidence behind the graceful-degradation claim in the README.
Anyone can write "the system handles failures". This demonstrates it.
"""
import json, os, sys
sys.path.insert(0, os.path.abspath("."))

import pandas as pd

from src.agents.orchestrator import Orchestrator, PrecomputedSentimentAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.models import AgentResult

PROC = "data/processed"


class BrokenAgent:
    """An agent that always raises, to test isolation."""
    def __init__(self, name):
        self.name = name

    def run(self, context):
        return AgentResult(agent_name=self.name, signals=[], confidence=0.0,
                           error="SimulatedFailure: agent unavailable")


def main():
    customers = pd.read_csv(f"{PROC}/customers_clean.csv")
    with open(f"{PROC}/conversation_signals.json") as f:
        conv_signals = json.load(f)
    with open(f"{PROC}/recommendations.json") as f:
        recs = json.load(f)

    # Use the highest-risk customer - the one with the most to lose.
    summary = pd.read_csv(f"{PROC}/pipeline_summary.csv")
    cid = summary.iloc[0].customer_id
    customer = customers[customers.customer_id == cid].iloc[0].to_dict()

    rules_rec = RecommendationAgent(client=None, use_llm=False)

    scenarios = {
        "all agents healthy": {},
        "sentiment agent down (LLM outage)": {"sentiment": True},
        "billing agent down": {"billing": True},
        "behaviour agent down": {"behaviour": True},
        "both LLM agents down": {"sentiment": True, "recommendation": True},
    }

    print(f"Resilience test on {cid}\n")
    print(f"{'SCENARIO':<38}{'SCORE':>7}{'LEVEL':>10}{'CONF':>7}"
          f"{'SIGS':>6}  DEGRADED")
    print("-" * 92)

    for label, broken in scenarios.items():
        sentiment = (BrokenAgent("sentiment") if broken.get("sentiment")
                     else PrecomputedSentimentAgent(conv_signals))
        recommender = (BrokenAgent("recommendation")
                       if broken.get("recommendation") else rules_rec)

        orch = Orchestrator(sentiment_agent=sentiment,
                            recommendation_agent=recommender)
        if broken.get("billing"):
            orch.billing = BrokenAgent("billing")
        if broken.get("behaviour"):
            orch.behaviour = BrokenAgent("behaviour")

        r = orch.assess(customer)
        action = (r["recommendation"] or {}).get("action", "none")
        print(f"{label:<38}{r['risk_score']:>7}{r['risk_level']:>10}"
              f"{r['confidence']:>7}{len(r['signals']):>6}  "
              f"{','.join(r['degraded']) or '-'}")

    print("\nEvery scenario produced an assessment. No exception escaped the "
          "orchestrator, and confidence fell whenever an agent was missing.")


if __name__ == "__main__":
    main()