"""
Central configuration for the Intelligent Customer Signal Detector.

Everything a business user might want to tune lives here: thresholds,
signal weights, and risk bands. Nothing is hard-coded inside the agents.

Why this matters: a reviewer will ask "what if inactivity should be 45 days
instead of 30?" The answer should be "change one line in config.py", not
"search the codebase for the number 30".
"""

import os

# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"

# --------------------------------------------------------------------------
# BEHAVIOURAL THRESHOLDS
# --------------------------------------------------------------------------
# Each entry maps a severity level to the value at which it triggers.
# Read as: "medium at 21+ days, high at 30+, critical at 60+".
THRESHOLDS = {
    "inactivity_days":        {"medium": 21,  "high": 30,  "critical": 60},
    "usage_decline_pct":      {"medium": -20, "high": -40, "critical": -70},
    "support_tickets":        {"medium": 4,   "high": 7,   "critical": 10},
    "unresolved_ratio":       {"medium": 0.30, "high": 0.50, "critical": 0.70},
    "resolution_hours":       {"medium": 48,  "high": 72,  "critical": 120},
    "days_since_payment":     {"medium": 40,  "high": 60,  "critical": 90},
    "contract_end_days":      {"medium": 60,  "high": 30,  "critical": 14},
}

# NPS is a 0-10 scale where LOW is bad, so it gets its own explicit mapping.
# 0-6 = Detractor, 7-8 = Passive, 9-10 = Promoter.
NPS_DETRACTOR = 6
NPS_SEVERE = 3

CSAT_LOW = 3.0
CSAT_SEVERE = 2.0

NEW_CUSTOMER_MONTHS = 3
NEW_CUSTOMER_MIN_LOGINS = 5

LOW_LOGIN_COUNT = 2          # logins in 30 days below this = LOGIN_DROP
ESTABLISHED_TENURE = 3       # months, before LOGIN_DROP applies

# --------------------------------------------------------------------------
# SIGNAL WEIGHTS  (used in Phase 6 scoring)
# --------------------------------------------------------------------------
# These are a documented starting hypothesis, NOT tuned against real outcomes.
# In production they would be calibrated by regressing signals against actual
# churn. Exposed here so the retention team can adjust them.
SIGNAL_WEIGHTS = {
    # Conversation signals (Phase 5) - highest weight, most direct evidence
    "CHURN_INTENT":          30,
    "COMPETITOR_MENTION":    20,
    "ESCALATION_LANGUAGE":   12,
    "REPEAT_ISSUE":          12,
    "HIGH_FRUSTRATION":      10,
    # Billing signals
    "PAYMENT_FAILURE":       18,
    "OVERDUE":               14,
    "CONTRACT_ENDING":        8,
    # Behavioural signals
    "DOWNGRADE":             15,
    "INACTIVITY":            14,
    "USAGE_DECLINE":         14,
    "LOGIN_DROP":            12,
    "LOW_NPS":               12,
    "UNRESOLVED_BACKLOG":    11,
    "SUPPORT_SPIKE":          8,
    "LOW_CSAT":               6,
    "SLOW_RESOLUTION":        5,
    "NEW_CUSTOMER_FRAGILE":   6,
    "NPS_SILENCE":            4,
    "NO_CONTACT":             3,
}

# Multiplier applied to the weight, based on how severe the signal is.
SEVERITY_MULTIPLIER = {
    "low":      0.5,
    "medium":   1.0,
    "high":     1.5,
    "critical": 2.0,
}

# --------------------------------------------------------------------------
# RISK BANDS
# --------------------------------------------------------------------------
RISK_BANDS = [
    (75, "Critical"),
    (55, "High"),
    (30, "Medium"),
    (0,  "Low"),
]


def risk_level(score):
    """Map a 0-100 score to a band name."""
    for threshold, label in RISK_BANDS:
        if score >= threshold:
            return label
    return "Low"