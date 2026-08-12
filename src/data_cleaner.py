"""
Data cleaning and validation for the Intelligent Customer Signal Detector.

Takes the messy raw CSVs and produces trustworthy processed CSVs, while
recording every change in a Data Quality Report.

Design principle: CLEANING changes data, VALIDATION checks it. We do both,
and we never change anything silently - every action is counted and printed.

Run:  python src/data_cleaner.py
"""

import os

import numpy as np
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
PROC_DIR = os.path.join("data", "processed")

# Valid ranges. Anything outside these is a data error, not a real value.
# Defined as config, not buried in code, so a reviewer can change them.
VALID_RANGES = {
    "nps_score": (0, 10),
    "csat_avg": (1.0, 5.0),
    "tenure_months": (0, 600),
    "last_login_days_ago": (0, 3650),
    "logins_last_30d": (0, 1000),
    "usage_pct_change_30d": (-100, 1000),
    "support_tickets_90d": (0, 500),
    "unresolved_tickets_90d": (0, 500),
    "avg_resolution_hours": (0, 8760),
    "failed_payments_90d": (0, 100),
    "days_since_last_payment": (0, 3650),
    "contract_end_days": (-365, 3650),
    "monthly_revenue": (0, 10_000_000),
}

# Columns whose missing values are IMPUTED (the gap is a logging failure).
IMPUTE_MEDIAN = ["avg_resolution_hours"]

# Columns whose missing values are KEPT (the gap is itself a signal).
KEEP_NULL = ["nps_score", "csat_avg"]

CATEGORICAL_TITLE = ["segment", "region"]
CATEGORICAL_STRIP = ["plan_tier", "plan_changed_last_90d"]


class QualityReport:
    """Collects every cleaning action so nothing happens silently."""

    def __init__(self, name):
        self.name = name
        self.rows_in = 0
        self.rows_out = 0
        self.actions = []

    def log(self, category, detail, count):
        if count > 0:
            self.actions.append((category, detail, count))

    def render(self):
        lines = []
        lines.append("=" * 72)
        lines.append(f"DATA QUALITY REPORT — {self.name}")
        lines.append("=" * 72)
        lines.append(f"Rows in:  {self.rows_in}")
        lines.append(f"Rows out: {self.rows_out}")
        if self.rows_in != self.rows_out:
            lines.append(f"Removed:  {self.rows_in - self.rows_out}")
        lines.append("")
        if not self.actions:
            lines.append("No issues found.")
        else:
            lines.append(f"{'ACTION':<20}{'DETAIL':<44}{'COUNT':>8}")
            lines.append("-" * 72)
            for cat, detail, count in self.actions:
                lines.append(f"{cat:<20}{detail:<44}{count:>8}")
        lines.append("=" * 72)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# CLEANING STEPS
# --------------------------------------------------------------------------

def _dedupe(df, key, report):
    """Remove duplicate rows, keeping the first occurrence."""
    before = len(df)
    df = df.drop_duplicates(subset=[key], keep="first").reset_index(drop=True)
    report.log("Deduplicate", f"duplicate {key} rows dropped", before - len(df))
    return df


def _normalise_text(df, report):
    """Fix casing and whitespace so categories group correctly."""
    for col in CATEGORICAL_TITLE:
        if col not in df.columns:
            continue
        original = df[col].copy()
        cleaned = df[col].astype("string").str.strip().str.title()
        # "Mid-Market" survives .title(); "SMB" would become "Smb", so restore
        cleaned = cleaned.replace({"Smb": "SMB"})
        changed = int((original.fillna("") != cleaned.fillna("")).sum())
        df[col] = cleaned
        report.log("Normalise text", f"{col}: casing/whitespace fixed", changed)

    for col in CATEGORICAL_STRIP:
        if col not in df.columns:
            continue
        original = df[col].copy()
        df[col] = df[col].astype("string").str.strip()
        changed = int((original.fillna("") != df[col].fillna("")).sum())
        report.log("Normalise text", f"{col}: whitespace stripped", changed)

    return df


def _coerce_types(df, report):
    """Convert columns to their proper types. errors='coerce' turns junk
    into NaT/NaN instead of raising, so one bad row can't kill the run."""
    if "signup_date" in df.columns:
        before_null = df["signup_date"].isna().sum()
        df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
        new_null = df["signup_date"].isna().sum() - before_null
        report.log("Type coercion", "signup_date -> datetime", len(df))
        report.log("Type coercion", "signup_date unparseable -> NaT", int(new_null))

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        report.log("Type coercion", "timestamp -> datetime", len(df))

    return df


def _validate_ranges(df, report):
    """Values outside a valid range are data errors. Set them to null rather
    than clipping - a NPS of 47 tells us nothing, so inventing 10 would be
    worse than admitting we don't know."""
    for col, (low, high) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        mask = df[col].notna() & ((df[col] < low) | (df[col] > high))
        n = int(mask.sum())
        if n:
            df.loc[mask, col] = np.nan
            report.log("Range validation",
                       f"{col}: outside [{low}, {high}] -> null", n)
    return df



def _repair_tenure(df, report):
    """Cross-field repair: tenure_months was invalidated by range validation,
    but signup_date holds the same information. Recompute rather than discard.

    This is better than imputing a median - we are not guessing, we are
    deriving the true value from another column that happens to be intact.
    """
    if not {"tenure_months", "signup_date"} <= set(df.columns):
        return df
    mask = df["tenure_months"].isna() & df["signup_date"].notna()
    n = int(mask.sum())
    if n:
        newest = pd.Timestamp("today").normalize()
        months = ((newest - df.loc[mask, "signup_date"]).dt.days / 30.44).round()
        df.loc[mask, "tenure_months"] = months.clip(lower=0)
        report.log("Cross-field repair",
                   "tenure_months recomputed from signup_date", n)
    still_missing = int(df["tenure_months"].isna().sum())
    if still_missing:
        df["tenure_months"] = df["tenure_months"].fillna(df["tenure_months"].median())
        report.log("Impute", "tenure_months: no signup_date, used median",
                   still_missing)
    df["tenure_months"] = df["tenure_months"].astype(int)
    return df


def _flag_missing(df, report):
    """Create explicit boolean flags BEFORE filling anything, so the fact
    that a value was missing survives into the signal layer."""
    for col in KEEP_NULL:
        if col not in df.columns:
            continue
        flag = f"{col}_missing"
        df[flag] = df[col].isna().astype(int)
        report.log("Flag missing", f"{flag} created", int(df[flag].sum()))
    return df


def _handle_missing(df, report):
    """Impute where the gap is a logging failure; keep null where the gap
    is itself meaningful customer behaviour."""
    for col in IMPUTE_MEDIAN:
        if col not in df.columns:
            continue
        n = int(df[col].isna().sum())
        if n:
            median = df[col].median()
            df[col] = df[col].fillna(median)
            report.log("Impute", f"{col}: filled with median {median:.1f}", n)

    if "region" in df.columns:
        n = int(df["region"].isna().sum())
        if n:
            df["region"] = df["region"].fillna("Unknown")
            report.log("Fill", "region: filled with 'Unknown'", n)

    for col in KEEP_NULL:
        if col in df.columns:
            n = int(df[col].isna().sum())
            report.log("Keep null", f"{col}: left null deliberately", n)

    return df


def _derive_features(df, report):
    """Columns computed from other columns. Kept separate from cleaning so
    it is obvious which fields are measured and which are calculated."""
    if {"unresolved_tickets_90d", "support_tickets_90d"} <= set(df.columns):
        df["unresolved_ratio"] = np.where(
            df["support_tickets_90d"] > 0,
            df["unresolved_tickets_90d"] / df["support_tickets_90d"],
            0.0,
        ).round(3)
        report.log("Derive", "unresolved_ratio computed", len(df))

    if "monthly_revenue" in df.columns:
        df["annual_value"] = (df["monthly_revenue"] * 12).round(2)
        report.log("Derive", "annual_value computed", len(df))

    return df


def _validate_referential(customers, conversations, report):
    """Every conversation must belong to a real customer. Orphan rows are
    dropped; customers with no conversations are KEPT (silence is a signal)."""
    valid_ids = set(customers["customer_id"])
    orphan_mask = ~conversations["customer_id"].isin(valid_ids)
    n_orphans = int(orphan_mask.sum())
    conversations = conversations[~orphan_mask].reset_index(drop=True)
    report.log("Referential", "orphan conversations dropped", n_orphans)

    no_contact = len(valid_ids - set(conversations["customer_id"]))
    report.log("Referential", "customers with zero conversations (kept)",
               no_contact)
    return conversations


def _validate_empty_text(df, report):
    """A blank transcript is useless to the LLM."""
    if "transcript" not in df.columns:
        return df
    mask = df["transcript"].isna() | (df["transcript"].astype(str).str.strip() == "")
    n = int(mask.sum())
    df = df[~mask].reset_index(drop=True)
    report.log("Content check", "empty transcripts dropped", n)
    return df


# --------------------------------------------------------------------------
# PUBLIC API
# --------------------------------------------------------------------------

def clean_customers(df):
    report = QualityReport("customers.csv")
    report.rows_in = len(df)

    df = df.copy()
    df = _dedupe(df, "customer_id", report)
    df = _normalise_text(df, report)
    df = _coerce_types(df, report)
    df = _validate_ranges(df, report)
    df = _repair_tenure(df, report)
    df = _flag_missing(df, report)
    df = _handle_missing(df, report)
    df = _derive_features(df, report)

    report.rows_out = len(df)
    return df, report


def clean_conversations(df):
    report = QualityReport("conversations.csv")
    report.rows_in = len(df)

    df = df.copy()
    df = _dedupe(df, "conversation_id", report)
    df = _coerce_types(df, report)
    df = _validate_empty_text(df, report)

    # Recency in days - needed for the weighting applied in aggregation.
    if "timestamp" in df.columns:
        newest = df["timestamp"].max()
        df["days_ago"] = (newest - df["timestamp"]).dt.days
        report.log("Derive", "days_ago computed for recency weighting", len(df))

    report.rows_out = len(df)
    return df, report


def run():
    os.makedirs(PROC_DIR, exist_ok=True)

    customers_raw = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"))
    conversations_raw = pd.read_csv(os.path.join(RAW_DIR, "conversations.csv"))

    customers, cust_report = clean_customers(customers_raw)
    conversations, conv_report = clean_conversations(conversations_raw)
    conversations = _validate_referential(customers, conversations, conv_report)

    customers.to_csv(os.path.join(PROC_DIR, "customers_clean.csv"), index=False)
    conversations.to_csv(os.path.join(PROC_DIR, "conversations_clean.csv"),
                         index=False)

    print(cust_report.render())
    print()
    print(conv_report.render())

    # Post-clean assertions. If any of these fail, downstream scoring is
    # untrustworthy, so we fail loudly here rather than silently later.
    assert customers["customer_id"].is_unique, "duplicate customer_id survived"
    assert customers["segment"].nunique() <= 4, "segment casing not normalised"
    assert customers["nps_score"].dropna().between(0, 10).all(), "NPS out of range"
    assert customers["csat_avg"].dropna().between(1, 5).all(), "CSAT out of range"
    assert (customers["tenure_months"].dropna() >= 0).all(), "negative tenure"
    print("\nAll post-clean assertions passed.")

    return customers, conversations


if __name__ == "__main__":
    run()