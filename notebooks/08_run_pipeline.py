"""
Run the full agentic pipeline end to end.

By default this replays cached LLM analysis, so it needs no API key and no
quota - which is exactly what you want when iterating on the dashboard or
recording a demo.

Usage:
    python notebooks/08_run_pipeline.py           # cached, no API calls
    python notebooks/08_run_pipeline.py --live    # call Gemini for the top N
    python notebooks/08_run_pipeline.py --serial  # disable parallelism (timing)
"""
import json, os, sys, time
sys.path.insert(0, os.path.abspath("."))

import pandas as pd

from src.agents.orchestrator import Orchestrator, PrecomputedSentimentAgent
from src.agents.recommendation_agent import RecommendationAgent

PROC = "data/processed"
LIVE = "--live" in sys.argv
PARALLEL = "--serial" not in sys.argv


class PrecomputedRecommendationAgent:
    """Replays recommendations captured in Phase 7."""
    name = "recommendation"

    def __init__(self, store):
        self.store = store or {}

    def run(self, context):
        from src.models import AgentResult
        cid = context["customer"]["customer_id"]
        rec = self.store.get(cid)
        result = AgentResult(agent_name=self.name, confidence=1.0)
        if rec is None:
            # Nothing cached - fall back to the deterministic path.
            return RecommendationAgent(client=None, use_llm=False).run(context)
        result.recommendation = rec
        return result


def load(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def main():
    customers = pd.read_csv(f"{PROC}/customers_clean.csv")
    conversations = pd.read_csv(f"{PROC}/conversations_clean.csv")
    conv_by_cust = {cid: g.to_dict("records")
                    for cid, g in conversations.groupby("customer_id")}

    conv_signals = load(f"{PROC}/conversation_signals.json", {})
    recs = load(f"{PROC}/recommendations.json", {})

    if LIVE:
        from src.agents.sentiment_agent import SentimentAgent
        sentiment = SentimentAgent()
        recommender = RecommendationAgent()
        print("LIVE mode - calling Gemini. This consumes quota.\n")
    else:
        sentiment = PrecomputedSentimentAgent(conv_signals)
        recommender = PrecomputedRecommendationAgent(recs)
        print(f"CACHED mode - replaying {len(conv_signals)} analyses "
              f"and {len(recs)} recommendations. No API calls.\n")

    orch = Orchestrator(sentiment_agent=sentiment,
                        recommendation_agent=recommender,
                        parallel=PARALLEL)

    print(f"Parallel stage-1 execution: {PARALLEL}")
    started = time.time()
    results = orch.assess_batch(customers.to_dict("records"), conv_by_cust)
    elapsed = time.time() - started

    with open(f"{PROC}/pipeline_output.json", "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame([{
        "customer_id": r["customer_id"], "name": r["name"],
        "segment": r["segment"], "monthly_revenue": r["monthly_revenue"],
        "risk_score": r["risk_score"], "risk_level": r["risk_level"],
        "confidence": r["confidence"], "n_signals": len(r["signals"]),
        "action": (r["recommendation"] or {}).get("action"),
        "priority": (r["recommendation"] or {}).get("priority"),
        "owner_team": (r["recommendation"] or {}).get("owner_team"),
        "degraded": ",".join(r["degraded"]),
        "latency_ms": r["total_latency_ms"],
    } for r in results]).sort_values("risk_score", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(f"{PROC}/pipeline_summary.csv", index=False)

    print(f"\nAssessed {len(results)} customers in {elapsed:.2f}s "
          f"({elapsed / len(results) * 1000:.1f}ms each)")
    print(f"Wrote {PROC}/pipeline_output.json and pipeline_summary.csv\n")

    # ------------------------------------------------------- agent trace
    print("AGENT TRACE (mean latency, total signals, errors)")
    agents = {}
    for r in results:
        for name, t in r["agent_trace"].items():
            a = agents.setdefault(name, {"lat": [], "sig": 0, "err": 0})
            a["lat"].append(t.get("latency_ms", 0))
            a["sig"] += t.get("signals", 0)
            a["err"] += 1 if t.get("error") else 0
    print(f"  {'AGENT':<16}{'MEAN ms':>9}{'SIGNALS':>10}{'ERRORS':>9}")
    for name, a in agents.items():
        print(f"  {name:<16}{sum(a['lat']) / len(a['lat']):>9.2f}"
              f"{a['sig']:>10}{a['err']:>9}")

    degraded = [r for r in results if r["degraded"]]
    print(f"\nDegraded assessments: {len(degraded)}")

    print("\nCONFIDENCE DISTRIBUTION")
    print(df.confidence.value_counts().sort_index(ascending=False).to_string())

    print("\nTOP 10")
    cols = ["rank", "customer_id", "segment", "risk_score", "risk_level",
            "confidence", "action", "owner_team"]
    print(df.head(10)[cols].to_string(index=False))


if __name__ == "__main__":
    main()