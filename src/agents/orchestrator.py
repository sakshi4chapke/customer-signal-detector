"""
Orchestrator - coordinates the five specialist agents.

    Stage 1 (parallel)   Behaviour · Billing · Sentiment
    Stage 2 (sequential) Risk Assessment      needs all signals
    Stage 3 (sequential) Recommendation       needs the score

Why an orchestrator rather than a script that calls functions in order:

  * PARALLELISM. Stages 1's three agents read different data and do not
    depend on each other, so they run concurrently.
  * FAILURE ISOLATION. One agent crashing must not lose the customer. The
    orchestrator collects what succeeded, lowers confidence, and continues.
  * TRACE. Every signal records which agent produced it, how long that agent
    took, and whether it errored. That trace is what makes the reasoning
    visible in the dashboard instead of the score arriving from nowhere.

Agents are injected rather than constructed internally, so the same
orchestrator runs live (calling Gemini) or offline (replaying cached
analysis) without changing a line of its logic.
"""

import time
from concurrent.futures import ThreadPoolExecutor

from src.agents.behaviour_agent import BehaviourAgent
from src.agents.billing_agent import BillingAgent
from src.agents.risk_agent import RiskAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.models import AgentResult, Signal
from src.signal_engine import dedupe

# How much of the risk picture each stage-1 agent is responsible for.
# Used to compute confidence as COVERAGE rather than as the worst agent:
# losing the billing view leaves an assessment weaker, not worthless.
# Behaviour carries the largest share because it is the only source that
# detects silent disengagement - the hardest and highest-value case.
AGENT_COVERAGE = {"behaviour": 0.45, "billing": 0.20, "sentiment": 0.35}


class PrecomputedSentimentAgent:
    """Replays Sentiment Agent output captured in an earlier run.

    The conversation analysis is the expensive, rate-limited part of the
    pipeline. Once captured it does not change, so the dashboard and the
    demo replay it instead of re-calling the API. Same output contract as
    the live agent, so the orchestrator cannot tell the difference.
    """

    name = "sentiment"

    def __init__(self, store):
        self.store = store or {}

    def run(self, context):
        cid = context.get("customer_id")
        record = self.store.get(cid, {})
        signals = [Signal(
            name=d["name"], severity=d["severity"], source=d["source"],
            evidence=d["evidence"], value=d.get("value"),
            recency_weight=d.get("recency_weight", 1.0),
        ) for d in record.get("signals", [])]

        confidence = 1.0
        if record.get("error"):
            confidence = 0.4
        elif record.get("skipped"):
            confidence = 0.6
        elif not record:
            confidence = 0.5

        return AgentResult(agent_name=self.name, signals=signals,
                           confidence=confidence, error=record.get("error"))


class Orchestrator:
    def __init__(self, sentiment_agent=None, recommendation_agent=None,
                 parallel=True):
        self.behaviour = BehaviourAgent()
        self.billing = BillingAgent()
        self.sentiment = sentiment_agent            # may be None
        self.risk = RiskAgent()
        self.recommendation = recommendation_agent  # may be None
        self.parallel = parallel

    # ------------------------------------------------------------ stage 1
    def _gather_signals(self, context):
        agents = [self.behaviour, self.billing]
        if self.sentiment is not None:
            agents.append(self.sentiment)

        if self.parallel and len(agents) > 1:
            with ThreadPoolExecutor(max_workers=len(agents)) as pool:
                results = list(pool.map(lambda a: a.run(context), agents))
        else:
            results = [a.run(context) for a in agents]

        return results

    # ------------------------------------------------------------- public
    def assess(self, customer, conversations=None):
        """Full assessment for one customer. Never raises."""
        started = time.perf_counter()
        context = {
            "customer": customer,
            "customer_id": customer["customer_id"],
            "conversations": conversations or [],
        }

        # --- Stage 1: independent specialists, run concurrently ----------
        results = self._gather_signals(context)

        all_signals = []
        trace = {}
        coverage = 0.0
        failures = []

        for r in results:
            all_signals.extend(r.signals)
            # An agent contributes its share of coverage, scaled by its own
            # confidence. A failed agent contributes nothing.
            coverage += AGENT_COVERAGE.get(r.agent_name, 0.0) * r.confidence
            trace[r.agent_name] = {
                "signals": len(r.signals),
                "confidence": r.confidence,
                "latency_ms": r.latency_ms,
                "error": r.error,
            }
            if r.error:
                failures.append(r.agent_name)

        signals = dedupe(all_signals)

        # Confidence = how much of the picture we actually saw. If the
        # sentiment agent is down we still have 65% coverage from behaviour
        # and billing, so the assessment is weaker but far from useless.
        # Normalised by the coverage that was theoretically available, so
        # running without a sentiment agent at all is not penalised twice.
        available = sum(AGENT_COVERAGE.get(a.name, 0.0)
                        for a in [self.behaviour, self.billing]
                        + ([self.sentiment] if self.sentiment else []))
        confidence = round(coverage / available, 2) if available else 0.0

        # --- Stage 2: scoring (needs every signal) -----------------------
        risk_result = self.risk.run({"signals": signals})
        trace["risk"] = {"latency_ms": risk_result.latency_ms,
                         "error": risk_result.error}
        assessment = getattr(risk_result, "assessment", None) or {
            "risk_score": 0.0, "raw_score": 0.0, "risk_level": "Low",
            "score_breakdown": [], "top_contributors": [],
        }

        # --- Stage 3: recommendation (needs the score) -------------------
        recommendation = None
        if self.recommendation is not None:
            rec_result = self.recommendation.run({
                "customer": customer,
                "assessment": assessment,
                "signals": [s.to_dict() for s in signals],
            })
            trace["recommendation"] = {"latency_ms": rec_result.latency_ms,
                                       "error": rec_result.error}
            recommendation = getattr(rec_result, "recommendation", None)
            if rec_result.error:
                failures.append("recommendation")

        total_ms = int((time.perf_counter() - started) * 1000)

        return {
            "customer_id": customer["customer_id"],
            "name": customer.get("name"),
            "segment": customer.get("segment"),
            "monthly_revenue": customer.get("monthly_revenue"),
            "risk_score": assessment["risk_score"],
            "risk_level": assessment["risk_level"],
            "confidence": confidence,
            "signals": [s.to_dict() for s in signals],
            "score_breakdown": assessment["score_breakdown"],
            "recommendation": recommendation,
            "agent_trace": trace,
            "degraded": failures,          # which agents did not contribute
            "total_latency_ms": total_ms,
        }

    def assess_batch(self, customers, conversations_by_customer=None,
                     max_workers=4, progress_every=50):
        """Assess many customers. Customers are independent, so they are
        processed concurrently too."""
        conversations_by_customer = conversations_by_customer or {}
        rows = list(customers)

        def one(c):
            return self.assess(c, conversations_by_customer.get(
                c["customer_id"], []))

        out = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for i, result in enumerate(pool.map(one, rows), start=1):
                out.append(result)
                if progress_every and i % progress_every == 0:
                    print(f"  assessed {i}/{len(rows)}")
        return out