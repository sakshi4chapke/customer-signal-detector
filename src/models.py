"""
Core data structures.

Every agent - rule-based or LLM-based - speaks in Signals and returns an
AgentResult. Because the shape is identical, the orchestrator can merge
results without special-casing any agent.

Dataclasses are used here rather than pydantic because these objects are
constructed internally by our own code, where the shape is guaranteed.
Pydantic is introduced in Phase 5, at the trust boundary where untrusted
LLM output enters the system and genuinely needs validation.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List

SEVERITY_ORDER = ["low", "medium", "high", "critical"]


@dataclass
class Signal:
    """A named, evidenced, severity-graded observation about one customer.

    The `evidence` field is the most important part. A signal without
    evidence is just a number; a signal with evidence can be shown to a
    retention agent who has 30 seconds before dialling.
    """
    name: str                       # e.g. "INACTIVITY"
    severity: str                   # low | medium | high | critical
    source: str                     # behaviour | billing | conversation
    evidence: str                   # human-readable justification
    value: Optional[float] = None   # the raw number behind it
    recency_weight: float = 1.0     # 1.0 for structured data; decays for text

    def __post_init__(self):
        if self.severity not in SEVERITY_ORDER:
            raise ValueError(
                f"{self.name}: severity '{self.severity}' is not one of "
                f"{SEVERITY_ORDER}"
            )

    @property
    def severity_rank(self):
        return SEVERITY_ORDER.index(self.severity)

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        return f"[{self.severity.upper():<8}] {self.name:<22} {self.evidence}"


@dataclass
class AgentResult:
    """What every agent returns. Uniform shape = easy orchestration."""
    agent_name: str
    signals: List[Signal] = field(default_factory=list)
    confidence: float = 1.0         # 1.0 = fully trusted (deterministic rules)
    error: Optional[str] = None
    latency_ms: int = 0

    @property
    def ok(self):
        return self.error is None

    def to_dict(self):
        return {
            "agent_name": self.agent_name,
            "signals": [s.to_dict() for s in self.signals],
            "confidence": self.confidence,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }