"""Run the rule agents over the cleaned data and inspect the signals."""
import sys, os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
from src.agents.behaviour_agent import BehaviourAgent
from src.agents.billing_agent import BillingAgent
from src.config import SIGNAL_WEIGHTS, SEVERITY_MULTIPLIER, risk_level

customers = pd.read_csv("data/processed/customers_clean.csv")
key = pd.read_csv("data/raw/answer_key.csv")

behaviour, billing = BehaviourAgent(), BillingAgent()

def naive_score(signals):
    raw = sum(SIGNAL_WEIGHTS.get(s.name, 0) * SEVERITY_MULTIPLIER[s.severity]
              for s in signals)
    return min(100, round(raw))

rows = []
for c in customers.to_dict("records"):
    ctx = {"customer": c}
    r1, r2 = behaviour.run(ctx), billing.run(ctx)
    sigs = r1.signals + r2.signals
    score = naive_score(sigs)
    rows.append({
        "customer_id": c["customer_id"], "name": c["name"],
        "segment": c["segment"], "mrr": c["monthly_revenue"],
        "n_signals": len(sigs), "score": score, "level": risk_level(score),
        "top_signals": ", ".join(s.name for s in
                                 sorted(sigs, key=lambda x: -x.severity_rank)[:3]),
        "errors": (r1.error or "") + (r2.error or ""),
    })

df = pd.DataFrame(rows).merge(key, on="customer_id")

print("AGENT ERRORS:", (df.errors != "").sum())
print("\nSIGNALS PER CUSTOMER");  print(df.n_signals.describe().round(1).to_string())
print("\nRISK DISTRIBUTION");     print(df.level.value_counts().to_string())
print("\nMEAN SCORE BY ARCHETYPE")
print(df.groupby("_archetype")[["score","churned_flag"]].mean().round(1).sort_values("score").to_string())
print("\nTOP 10 AT RISK")
print(df.nlargest(10,"score")[["customer_id","segment","score","level","_archetype","churned_flag","top_signals"]].to_string(index=False))
print("\nPRECISION@10:", f"{df.nlargest(10,'score').churned_flag.mean():.0%}")
print("PRECISION@20:", f"{df.nlargest(20,'score').churned_flag.mean():.0%}")