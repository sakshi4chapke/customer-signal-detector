"""
Generate retention recommendations for the highest-risk customers.

The LLM is spent only where it changes the answer: nobody writes a retention
script for a healthy customer. Everyone else gets the deterministic
fallback, so no customer is ever left without an action.

Usage:
    python notebooks/07_recommendations.py          # top 30 via LLM
    python notebooks/07_recommendations.py 10       # top 10 via LLM
    python notebooks/07_recommendations.py 0        # rules only, no API calls
"""
import json, os, sys
sys.path.insert(0, os.path.abspath("."))

import pandas as pd

from src.agents.recommendation_agent import RecommendationAgent
from src.llm_client import LLMClient, LLMUnavailable, LLMFatalError

PROC = "data/processed"
TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def main():
    scored = pd.read_csv(f"{PROC}/scored_customers.csv")
    customers = pd.read_csv(f"{PROC}/customers_clean.csv").set_index("customer_id")
    with open(f"{PROC}/assessments.json") as f:
        assessments = json.load(f)

    client = None
    if TOP_N > 0:
        try:
            client = LLMClient()
            ok, msg = client.healthcheck()
            if not ok:
                print(f"Healthcheck failed: {msg}\nUsing rules-based "
                      f"recommendations for all customers.\n")
                client = None
            else:
                print(f"Healthcheck OK: {msg}\n")
        except LLMUnavailable as exc:
            print(f"LLM unavailable ({exc}); rules-based for all.\n")

    llm_agent = RecommendationAgent(client=client) if client else None
    rules_agent = RecommendationAgent(client=None, use_llm=False)

    top_ids = set(scored.nlargest(TOP_N, "risk_score").customer_id) if TOP_N else set()
    print(f"LLM briefings for top {len(top_ids)} · "
          f"rules-based for the remaining {len(scored) - len(top_ids)}\n")

    out = {}
    llm_used = 0
    for _, row in scored.iterrows():
        cid = row.customer_id
        a = assessments[cid]
        context = {
            "customer": customers.loc[cid].to_dict() | {"customer_id": cid},
            "assessment": a,
            "signals": a["signals"],
        }

        agent = llm_agent if (cid in top_ids and llm_agent) else rules_agent
        try:
            result = agent.run(context)
        except LLMFatalError as exc:
            print(f"Quota exhausted after {llm_used} briefings: "
                  f"{str(exc)[:100]}\nFalling back to rules for the rest.\n")
            llm_agent = None
            result = rules_agent.run(context)

        rec = result.recommendation
        if rec["generated_by"] == "llm":
            llm_used += 1
        out[cid] = rec

    with open(f"{PROC}/recommendations.json", "w") as f:
        json.dump(out, f, indent=2)

    # Merge into the priority list so the CSV is self-contained.
    scored["action"] = scored.customer_id.map(lambda c: out[c]["action"])
    scored["priority"] = scored.customer_id.map(lambda c: out[c]["priority"])
    scored["owner_team"] = scored.customer_id.map(lambda c: out[c]["owner_team"])
    scored["explanation"] = scored.customer_id.map(lambda c: out[c]["explanation"])
    scored["rec_source"] = scored.customer_id.map(lambda c: out[c]["generated_by"])
    scored.to_csv(f"{PROC}/scored_customers.csv", index=False)

    print(f"Wrote {PROC}/recommendations.json")
    print(f"LLM-generated briefings: {llm_used}\n")

    print("ACTION DISTRIBUTION")
    print(scored.action.value_counts().to_string())

    print("\nWORKLOAD BY TEAM (High and Critical only)")
    urgent = scored[scored.risk_level.isin(["High", "Critical"])]
    print(urgent.groupby("owner_team").size().sort_values(ascending=False).to_string())

    print("\n" + "=" * 78)
    print("SAMPLE BRIEFINGS")
    print("=" * 78)
    for cid in scored.nlargest(3, "risk_score").customer_id:
        r, row = out[cid], scored[scored.customer_id == cid].iloc[0]
        print(f"\n{cid} · {row['name']} · {row.segment} · "
              f"{row.monthly_revenue:,.0f}/mo")
        print(f"  Risk {row.risk_score} ({row.risk_level}) · "
              f"{r['action']} · {r['priority']} · {r['owner_team']} · "
              f"SLA {r['sla_hours']}h · via {r['generated_by']}")
        print(f"\n  {r['explanation']}\n")
        for p in r["talking_points"]:
            print(f"   - {p}")
    print()


if __name__ == "__main__":
    main()