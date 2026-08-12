import sys, os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
from src.agents.behaviour_agent import BehaviourAgent
from src.agents.billing_agent import BillingAgent

customers = pd.read_csv("data/processed/customers_clean.csv")
key = pd.read_csv("data/raw/answer_key.csv")

# Pick one quiet drifter and one loud-but-stable customer
merged = customers.merge(key, on="customer_id")
targets = [
    merged[merged._archetype == "quiet_drifter"].iloc[0],
    merged[merged._archetype == "loud_but_stable"].iloc[0],
]

for row in targets:
    c = row.to_dict()
    print("=" * 70)
    print(f"{c['customer_id']}  {c['name']}  ({c['_archetype']})")
    print(f"  {c['last_login_days_ago']}d since login · "
          f"usage {c['usage_pct_change_30d']:+.0f}% · "
          f"{c['support_tickets_90d']} tickets · "
          f"NPS {c['nps_score']}")
    print("=" * 70)
    for agent in (BehaviourAgent(), BillingAgent()):
        result = agent.run({"customer": c})
        print(f"\n  {agent.name.upper()} AGENT  "
              f"({len(result.signals)} signals, {result.latency_ms}ms)")
        for s in result.signals:
            print(f"    {s}")
    print()