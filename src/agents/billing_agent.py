"""
Billing Agent - reads payment history and contract timing.

Kept separate from the Behaviour Agent for three reasons:
  1. In a real deployment these come from a different system (finance vs
     product analytics), often with different refresh rates.
  2. Billing problems need a different remediation team, so the signal
     source determines who gets the ticket.
  3. Separate agents make the reasoning trace legible in the UI - the user
     can see that a flag came from Billing, not from usage data.
"""

from src.agents.base_agent import BaseAgent
from src.models import AgentResult, Signal

SOURCE = "billing"


def _get(c, key, default=None):
    v = c.get(key, default)
    if v is None:
        return default
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return default
    except Exception:
        pass
    return v


def rule_payment_failure(c, grade):
    failed = _get(c, "failed_payments_90d", 0)
    if not failed:
        return None
    sev = "critical" if failed >= 2 else "high"
    return Signal(
        name="PAYMENT_FAILURE", severity=sev, source=SOURCE, value=failed,
        evidence=(f"{int(failed)} failed payment attempt"
                  f"{'s' if failed > 1 else ''} in 90 days"),
    )


def rule_overdue(c, grade):
    days = _get(c, "days_since_last_payment")
    sev = grade(days, "days_since_payment")
    if not sev:
        return None
    return Signal(
        name="OVERDUE", severity=sev, source=SOURCE, value=days,
        evidence=f"No successful payment for {int(days)} days",
    )


def rule_contract_ending(c, grade):
    """Reverse threshold: FEWER days remaining is worse."""
    days = _get(c, "contract_end_days")
    sev = grade(days, "contract_end_days", reverse=True)
    if not sev:
        return None
    return Signal(
        name="CONTRACT_ENDING", severity=sev, source=SOURCE, value=days,
        evidence=f"Contract renewal in {int(days)} days",
    )


RULES = [rule_payment_failure, rule_overdue, rule_contract_ending]


class BillingAgent(BaseAgent):
    name = "billing"

    def analyse(self, context):
        customer = context["customer"]
        signals = [s for s in (rule(customer, self.grade) for rule in RULES)
                   if s is not None]
        return AgentResult(agent_name=self.name, signals=signals,
                           confidence=1.0)