"""
Base class every agent inherits from.

An agent is four things:
  1. a defined role         (what slice of the problem it owns)
  2. its own input slice    (which data it reads)
  3. its own decision logic (rules, or an LLM call)
  4. a structured output    (always an AgentResult)

Not every agent needs an LLM. Behaviour and Billing are deterministic rule
agents; Sentiment and Recommendation (Phase 5, 8) are LLM agents. What makes
them all "agents" is the contract, not the brain.
"""

import time
from abc import ABC, abstractmethod

from src.models import AgentResult
from src.config import THRESHOLDS


class BaseAgent(ABC):
    name = "base"

    @abstractmethod
    def analyse(self, context: dict) -> AgentResult:
        """Take a context dict, return an AgentResult. Must not raise."""
        raise NotImplementedError

    def run(self, context: dict) -> AgentResult:
        """Wrapper that times the agent and converts any crash into a
        recorded error rather than an exception. One failing agent must
        never take down the whole assessment."""
        start = time.perf_counter()
        try:
            result = self.analyse(context)
        except Exception as exc:                      # noqa: BLE001
            result = AgentResult(
                agent_name=self.name,
                signals=[],
                confidence=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        result.latency_ms = int((time.perf_counter() - start) * 1000)
        return result

    # ---------------------------------------------------------------- utils

    @staticmethod
    def grade(value, key, reverse=False):
        """Turn a raw number into a severity string using THRESHOLDS.

        reverse=False : higher is worse (e.g. days inactive)
        reverse=True  : lower is worse  (e.g. usage change %, which is
                        negative when declining)

        Returns None when the value doesn't reach even the medium threshold,
        which means "no signal" - the customer is fine on this dimension.
        """
        if value is None:
            return None
        t = THRESHOLDS[key]
        if reverse:
            if value <= t["critical"]:
                return "critical"
            if value <= t["high"]:
                return "high"
            if value <= t["medium"]:
                return "medium"
        else:
            if value >= t["critical"]:
                return "critical"
            if value >= t["high"]:
                return "high"
            if value >= t["medium"]:
                return "medium"
        return None