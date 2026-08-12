"""
Score every customer and write the ranked priority list.

Output: data/processed/scored_customers.csv  (the product)
        data/processed/assessments.json      (full detail + agent trace)
"""
import json, os, sys
sys.path.insert(0, os.path.abspath("."))

import pandas as pd

from src.signal_engine import build_customer_signals, load_conversation_signals
from src.agents.risk_agent import RiskAgent
from src.config import risk_level

PROC = "data/processed"


def main():
    customers = pd.read_csv(f"{PROC}/customers_clean.csv")
    conv_signals = load_conversation_signals()
    if not conv_signals:
        print("No conversation signals found - scoring on rules only.\n")

    merged = build_customer_signals(customers, conv_signals)
    risk = RiskAgent()

    rows, assessments = [], {}
    for row in customers.to_dict("records"):
        cid = row["customer_id"]
        data = merged[cid]
        result = risk.run({"signals": data["signals"]})
        a = result.assessment

        rows.append({
            "customer_id": cid,
            "name": row["name"],
            "segment": row["segment"],
            "plan_tier": row["plan_tier"],
            "monthly_revenue": row["monthly_revenue"],
            "annual_value": row["annual_value"],
            "risk_score": a["risk_score"],
            "raw_score": a["raw_score"],
            "risk_level": a["risk_level"],
            "confidence": data["confidence"],
            "n_signals": len(data["signals"]),
            "top_signals": " | ".join(a["top_contributors"]),
            "sources": ",".join(sorted({s.source for s in data["signals"]})),
        })

        assessments[cid] = {
            "risk_score": a["risk_score"],
            "risk_level": a["risk_level"],
            "confidence": data["confidence"],
            "signals": [s.to_dict() for s in data["signals"]],
            "score_breakdown": a["score_breakdown"],
            "agent_trace": data["agent_trace"],
        }

    df = pd.DataFrame(rows).sort_values("risk_score", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    df["value_at_risk"] = (df.risk_score / 100 * df.annual_value).round(0)

    df.to_csv(f"{PROC}/scored_customers.csv", index=False)
    with open(f"{PROC}/assessments.json", "w") as f:
        json.dump(assessments, f, indent=2)

    # ------------------------------------------------------------ reporting
    print(f"Wrote {PROC}/scored_customers.csv and assessments.json\n")

    print("SCORE DISTRIBUTION")
    print(df.risk_score.describe().round(1).to_string())
    print(f"\nExact ties at the top: "
          f"{(df.risk_score == df.risk_score.max()).sum()}")

    print("\nRISK BANDS")
    order = ["Critical", "High", "Medium", "Low"]
    counts = df.risk_level.value_counts().reindex(order).fillna(0).astype(int)
    for lvl, n in counts.items():
        var = df[df.risk_level == lvl].value_at_risk.sum()
        print(f"  {lvl:<9} {n:>3} customers   value at risk {var:>12,.0f}")

    print("\nTOP 15 PRIORITY LIST")
    cols = ["rank", "customer_id", "segment", "risk_score", "risk_level",
            "confidence", "monthly_revenue", "top_signals"]
    print(df.head(15)[cols].to_string(index=False))

    # ---------------------------------------------------------- evaluation
    key = pd.read_csv("data/raw/answer_key.csv")
    ev = df.merge(key, on="customer_id")

    print("\nMEAN SCORE BY ARCHETYPE")
    print(ev.groupby("_archetype")[["risk_score", "churned_flag"]]
            .mean().round(1).sort_values("risk_score").to_string())

    print("\nPRECISION AT K")
    for k in (5, 10, 20, 30, 50):
        print(f"  P@{k:<3} {ev.nlargest(k, 'risk_score').churned_flag.mean():.0%}")

    flagged = ev[ev.risk_level.isin(["High", "Critical"])]
    tp = int(flagged.churned_flag.sum())
    fp = len(flagged) - tp
    fn = int(ev.churned_flag.sum()) - tp
    tn = len(ev) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    print("\nCONFUSION MATRIX  (flagged = High or Critical)")
    print(f"                     churned   retained")
    print(f"  flagged            {tp:>7}   {fp:>8}")
    print(f"  not flagged        {fn:>7}   {tn:>8}")
    print(f"\n  Precision {precision:.0%} · Recall {recall:.0%} · F1 {f1:.2f}")


if __name__ == "__main__":
    main()