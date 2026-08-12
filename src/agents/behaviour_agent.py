"""
Behaviour Agent - reads engagement, usage, support, and satisfaction data.

Deterministic. No LLM. Same input always produces the same signals, which is
what makes the risk score reproducible and auditable.

Each rule is a small function returning either a Signal or None. Keeping them
separate means you can read, test, and change one rule without touching the
others - and the rule registry doubles as documentation of what the agent
looks for.
"""

from src.agents.base_agent import BaseAgent
from src.models import AgentResult, Signal
from src.config import (
    NPS_DETRACTOR, NPS_SEVERE, CSAT_LOW, CSAT_SEVERE,
    NEW_CUSTOMER_MONTHS, NEW_CUSTOMER_MIN_LOGINS,
    LOW_LOGIN_COUNT, ESTABLISHED_TENURE,
)

SOURCE = "behaviour"


def _get(c, key, default=None):
    """Safe accessor - a missing column must never crash an agent."""
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


# ------------------------------------------------------------------- rules

def rule_inactivity(c, grade):
    days = _get(c, "last_login_days_ago")
    sev = grade(days, "inactivity_days")
    if not sev:
        return None
    return Signal(
        name="INACTIVITY", severity=sev, source=SOURCE, value=days,
        evidence=f"No login for {int(days)} days",
    )


def rule_login_drop(c, grade):
    logins = _get(c, "logins_last_30d")
    tenure = _get(c, "tenure_months", 0)
    if logins is None or tenure < ESTABLISHED_TENURE:
        return None
    if logins > LOW_LOGIN_COUNT:
        return None
    sev = "high" if logins == 0 else "medium"
    return Signal(
        name="LOGIN_DROP", severity=sev, source=SOURCE, value=logins,
        evidence=(f"Only {int(logins)} logins in the last 30 days "
                  f"despite {int(tenure)} months tenure"),
    )


def rule_usage_decline(c, grade):
    pct = _get(c, "usage_pct_change_30d")
    sev = grade(pct, "usage_decline_pct", reverse=True)
    if not sev:
        return None
    return Signal(
        name="USAGE_DECLINE", severity=sev, source=SOURCE, value=pct,
        evidence=f"Usage down {abs(pct):.0f}% versus the previous 30 days",
    )


def rule_support_spike(c, grade):
    tickets = _get(c, "support_tickets_90d")
    sev = grade(tickets, "support_tickets")
    if not sev:
        return None
    return Signal(
        name="SUPPORT_SPIKE", severity=sev, source=SOURCE, value=tickets,
        evidence=f"{int(tickets)} support tickets raised in 90 days",
    )


def rule_unresolved_backlog(c, grade):
    """Ticket volume alone cannot separate a vocal loyal customer from a
    departing one - both complain. Resolution RATE can."""
    ratio = _get(c, "unresolved_ratio")
    unresolved = _get(c, "unresolved_tickets_90d", 0)
    if ratio is None or unresolved == 0:
        return None
    sev = grade(ratio, "unresolved_ratio")
    if not sev:
        return None
    total = _get(c, "support_tickets_90d", 0)
    return Signal(
        name="UNRESOLVED_BACKLOG", severity=sev, source=SOURCE, value=ratio,
        evidence=(f"{int(unresolved)} of {int(total)} tickets still unresolved "
                  f"({ratio:.0%})"),
    )


def rule_slow_resolution(c, grade):
    hours = _get(c, "avg_resolution_hours")
    sev = grade(hours, "resolution_hours")
    if not sev:
        return None
    return Signal(
        name="SLOW_RESOLUTION", severity=sev, source=SOURCE, value=hours,
        evidence=f"Average resolution time {hours:.0f} hours",
    )


def rule_low_nps(c, grade):
    nps = _get(c, "nps_score")
    if nps is None or nps > NPS_DETRACTOR:
        return None
    sev = "critical" if nps <= NPS_SEVERE else "high"
    return Signal(
        name="LOW_NPS", severity=sev, source=SOURCE, value=nps,
        evidence=f"NPS of {int(nps)} - detractor (0-6 range)",
    )


def rule_nps_silence(c, grade):
    """Never answering the survey is not neutral. Non-response correlates
    with disengagement, which is why we kept the null instead of imputing."""
    if not _get(c, "nps_score_missing", 0):
        return None
    return Signal(
        name="NPS_SILENCE", severity="low", source=SOURCE, value=None,
        evidence="No NPS response on record - survey never completed",
    )


def rule_low_csat(c, grade):
    csat = _get(c, "csat_avg")
    if csat is None or csat >= CSAT_LOW:
        return None
    sev = "high" if csat < CSAT_SEVERE else "medium"
    return Signal(
        name="LOW_CSAT", severity=sev, source=SOURCE, value=csat,
        evidence=f"Average CSAT {csat:.1f} out of 5",
    )


def rule_downgrade(c, grade):
    change = str(_get(c, "plan_changed_last_90d", "none")).lower()
    if change != "downgrade":
        return None
    return Signal(
        name="DOWNGRADE", severity="high", source=SOURCE, value=None,
        evidence="Plan downgraded within the last 90 days",
    )


def rule_new_customer_fragile(c, grade):
    tenure = _get(c, "tenure_months")
    logins = _get(c, "logins_last_30d")
    if tenure is None or logins is None:
        return None
    if tenure >= NEW_CUSTOMER_MONTHS or logins >= NEW_CUSTOMER_MIN_LOGINS:
        return None
    return Signal(
        name="NEW_CUSTOMER_FRAGILE", severity="medium", source=SOURCE,
        value=tenure,
        evidence=(f"New customer ({int(tenure)} months) with only "
                  f"{int(logins)} logins - onboarding may have failed"),
    )


RULES = [
    rule_inactivity,
    rule_login_drop,
    rule_usage_decline,
    rule_support_spike,
    rule_unresolved_backlog,
    rule_slow_resolution,
    rule_low_nps,
    rule_nps_silence,
    rule_low_csat,
    rule_downgrade,
    rule_new_customer_fragile,
]


class BehaviourAgent(BaseAgent):
    name = "behaviour"

    def analyse(self, context):
        customer = context["customer"]
        signals = [s for s in (rule(customer, self.grade) for rule in RULES)
                   if s is not None]
        return AgentResult(agent_name=self.name, signals=signals,
                           confidence=1.0)