"""
Run the Sentiment Agent across every customer and save the results.

Because the LLM client caches by prompt hash, re-running this is nearly
instant and costs no quota. That matters: you will run it many times while
building the dashboard.

Usage:
    python notebooks/05_run_sentiment.py            # all customers
    python notebooks/05_run_sentiment.py 20         # first 20 only (testing)
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

import pandas as pd

from src.agents.sentiment_agent import SentimentAgent
from src.llm_client import LLMClient, LLMUnavailable, LLMFatalError

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None
OUT = "data/processed/conversation_signals.json"


def main():
    customers = pd.read_csv("data/processed/customers_clean.csv")
    conversations = pd.read_csv("data/processed/conversations_clean.csv")

    if LIMIT:
        customers = customers.head(LIMIT)
        print(f"Limiting to first {LIMIT} customers.\n")

    try:
        client = LLMClient()
    except LLMUnavailable as exc:
        print(f"LLM unavailable: {exc}")
        print("Falling back to rules-only. Set GEMINI_API_KEY in .env to enable.")
        return

    # Fail fast: prove the model name and key work before starting a run
    # that would otherwise fail identically 169 times.
    ok, message = client.healthcheck()
    if not ok:
        print(f"HEALTHCHECK FAILED\n  {message}\n")
        print("Fix the model name or key in .env before running. "
              "Current model: " + client.model)
        return
    print(f"Healthcheck OK: {message}\n")

    agent = SentimentAgent(client=client)
    by_customer = {cid: g.to_dict("records")
                   for cid, g in conversations.groupby("customer_id")}

    results = {}
    errors = 0
    consecutive = 0
    started = time.time()

    for i, cid in enumerate(customers["customer_id"], start=1):
        context = {"conversations": by_customer.get(cid, [])}
        try:
            result = agent.run(context)
        except LLMFatalError as exc:
            print(f"\nFATAL after {i} customers: {exc}")
            print("Aborting - this error cannot be fixed by retrying.")
            break

        if result.error:
            errors += 1
            if errors <= 3:                       # show the first few, not 169
                print(f"    error on {cid}: {result.error[:140]}")
            consecutive += 1
            if consecutive >= 10:
                print(f"\nSTOPPING: 10 consecutive failures. "
                      f"Last error: {result.error[:200]}")
                break
        else:
            consecutive = 0

        results[cid] = result.to_dict()

        if i % 10 == 0 or i == len(customers):
            elapsed = time.time() - started
            print(f"  {i:>3}/{len(customers)} customers · "
                  f"{elapsed:>5.1f}s · {client.report()}")

    os.makedirs("data/processed", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------- summary
    all_signals = [s for r in results.values() for s in r["signals"]]
    counts = pd.Series([s["name"] for s in all_signals]).value_counts()

    print(f"\nWrote {OUT}")
    print(f"Customers processed : {len(results)}")
    print(f"Agent errors        : {errors}")
    print(f"Total signals       : {len(all_signals)}")
    print(f"\nSIGNAL FREQUENCY\n{counts.to_string()}")
    print(f"\n{client.report()}")

    # Show a few real examples - this is what the dashboard will display.
    print("\nSAMPLE SIGNALS WITH EVIDENCE")
    shown = 0
    for cid, r in results.items():
        interesting = [s for s in r["signals"]
                       if s["name"] in ("CHURN_INTENT", "COMPETITOR_MENTION")]
        if interesting and shown < 3:
            print(f"\n  {cid}")
            for s in r["signals"]:
                print(f"    [{s['severity'].upper():<8}] {s['name']:<20} "
                      f"{s['evidence'][:90]}")
            shown += 1


if __name__ == "__main__":
    main()