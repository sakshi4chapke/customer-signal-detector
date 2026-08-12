"""
Synthetic dataset generator for the Intelligent Customer Signal Detector.

Creates two CSVs in data/raw/:
  - customers.csv      : one row per customer (structured behavioural data)
  - conversations.csv  : many rows per customer (unstructured transcripts)

Design principle: we PLANT the patterns before we detect them.
Six archetypes are generated with deliberately different signal profiles so
that the scoring engine has something real to find. `churned_flag` is the
hidden answer key used ONLY for evaluation in Phase 12 - never as an input
to the risk score.

Run:  python src/data_generator.py
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Fixed seed = same data every run = reproducible demo and reproducible tests.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_CUSTOMERS = 200
OUT_DIR = os.path.join("data", "raw")
TODAY = datetime(2026, 8, 12)

# --------------------------------------------------------------------------
# 1. ARCHETYPE DEFINITIONS
# --------------------------------------------------------------------------
# Each archetype is a "recipe" describing the RANGE each field should fall in.
# Tuples are (low, high) and we draw a random value inside that range.

ARCHETYPES = {
    "healthy_loyal": {
        "weight": 0.35,
        "tenure_months": (18, 72),
        "last_login_days_ago": (0, 4),
        "logins_last_30d": (15, 60),
        "usage_pct_change_30d": (-8, 25),
        "support_tickets_90d": (0, 2),
        "unresolved_frac": (0.0, 0.10),
        "avg_resolution_hours": (2, 14),
        "failed_payments_90d": (0, 0),
        "days_since_last_payment": (1, 30),
        "nps_score": (8, 10),
        "csat_avg": (4.2, 5.0),
        "plan_change": ["none", "none", "none", "upgrade"],
        "contract_end_days": (120, 400),
        "n_conversations": (0, 2),
        "churn_prob": 0.02,
    },
    "quiet_drifter": {
        # THE STAR OF THE DEMO.
        # Numbers scream, text says nothing. Classic silent churn.
        "weight": 0.15,
        "tenure_months": (12, 48),
        "last_login_days_ago": (32, 75),
        "logins_last_30d": (0, 2),
        "usage_pct_change_30d": (-85, -45),
        "support_tickets_90d": (0, 1),
        "unresolved_frac": (0.0, 0.20),
        "avg_resolution_hours": (4, 20),
        "failed_payments_90d": (0, 0),
        "days_since_last_payment": (1, 35),
        "nps_score": (None, None),          # never responded to the survey
        "csat_avg": (None, None),
        "plan_change": ["none", "none", "downgrade"],
        "contract_end_days": (20, 120),
        "n_conversations": (0, 1),
        "churn_prob": 0.68,
    },
    "loud_but_stable": {
        # THE CONTROL CASE.
        # Text screams, numbers are fine. Engaged, not leaving.
        "weight": 0.15,
        "tenure_months": (10, 60),
        "last_login_days_ago": (0, 3),
        "logins_last_30d": (18, 55),
        "usage_pct_change_30d": (-5, 20),
        "support_tickets_90d": (4, 9),
        "unresolved_frac": (0.0, 0.25),
        "avg_resolution_hours": (10, 40),
        "failed_payments_90d": (0, 0),
        "days_since_last_payment": (1, 28),
        "nps_score": (6, 8),
        "csat_avg": (3.0, 4.0),
        "plan_change": ["none", "none", "upgrade"],
        "contract_end_days": (100, 365),
        "n_conversations": (3, 6),
        "churn_prob": 0.10,
    },
    "billing_wounded": {
        "weight": 0.15,
        "tenure_months": (6, 40),
        "last_login_days_ago": (1, 18),
        "logins_last_30d": (4, 20),
        "usage_pct_change_30d": (-30, 5),
        "support_tickets_90d": (2, 6),
        "unresolved_frac": (0.30, 0.60),
        "avg_resolution_hours": (24, 90),
        "failed_payments_90d": (1, 3),
        "days_since_last_payment": (38, 85),
        "nps_score": (2, 6),
        "csat_avg": (1.8, 3.2),
        "plan_change": ["none", "downgrade"],
        "contract_end_days": (30, 200),
        "n_conversations": (2, 5),
        "churn_prob": 0.52,
    },
    "escalating_churner": {
        # THE OBVIOUS ONE. Everything is on fire.
        "weight": 0.12,
        "tenure_months": (4, 30),
        "last_login_days_ago": (10, 45),
        "logins_last_30d": (0, 5),
        "usage_pct_change_30d": (-95, -50),
        "support_tickets_90d": (5, 12),
        "unresolved_frac": (0.50, 0.85),
        "avg_resolution_hours": (40, 140),
        "failed_payments_90d": (0, 2),
        "days_since_last_payment": (10, 60),
        "nps_score": (0, 3),
        "csat_avg": (1.0, 2.4),
        "plan_change": ["downgrade", "downgrade", "none"],
        "contract_end_days": (5, 60),
        "n_conversations": (3, 7),
        "churn_prob": 0.88,
    },
    "new_fragile": {
        "weight": 0.08,
        "tenure_months": (1, 3),
        "last_login_days_ago": (5, 25),
        "logins_last_30d": (1, 6),
        "usage_pct_change_30d": (-40, 10),
        "support_tickets_90d": (1, 4),
        "unresolved_frac": (0.20, 0.50),
        "avg_resolution_hours": (12, 50),
        "failed_payments_90d": (0, 1),
        "days_since_last_payment": (1, 40),
        "nps_score": (4, 7),
        "csat_avg": (2.5, 3.8),
        "plan_change": ["none"],
        "contract_end_days": (250, 340),
        "n_conversations": (1, 4),
        "churn_prob": 0.35,
    },
}

SEGMENTS = ["Enterprise", "Mid-Market", "SMB", "Startup"]
PLAN_TIERS = ["Basic", "Standard", "Premium", "Enterprise"]
REGIONS = ["North", "South", "East", "West", "Central"]
CHANNELS = ["chat", "email", "phone_transcript"]

FIRST_NAMES = ["Aarav", "Priya", "Rohan", "Sneha", "Vikram", "Ananya", "Karan",
               "Meera", "Arjun", "Divya", "Rahul", "Kavya", "Siddharth", "Neha",
               "Amit", "Pooja", "Nikhil", "Riya", "Sanjay", "Tara", "James",
               "Emily", "Michael", "Sarah", "David", "Laura", "Daniel", "Fatima",
               "Omar", "Chen", "Yuki", "Carlos", "Sofia", "Marcus", "Elena"]
LAST_NAMES = ["Sharma", "Patel", "Reddy", "Nair", "Iyer", "Gupta", "Singh",
              "Mehta", "Joshi", "Rao", "Desai", "Kapoor", "Bose", "Malhotra",
              "Smith", "Johnson", "Williams", "Brown", "Garcia", "Miller",
              "Khan", "Ahmed", "Lee", "Wang", "Rossi", "Muller", "Silva"]


# --------------------------------------------------------------------------
# 2. TRANSCRIPT TEMPLATES
# --------------------------------------------------------------------------
# Real customers never say "I am at risk of churning". They complain about
# specific things. Templates carry the signals we want the LLM to find:
# churn intent, competitor mentions, repeat issues, escalation language.
# Deliberate typos are included - real chat logs are messy.

TRANSCRIPTS = {
    "healthy_loyal": [
        "Customer: Hi, quick question - how do I add a second admin user?\n"
        "Agent: Happy to help! Go to Settings > Team > Invite.\n"
        "Customer: Found it, thanks. Works great as always.",

        "Customer: Just wanted to say the new dashboard update is really good.\n"
        "Agent: Thank you, I'll pass that to the product team!\n"
        "Customer: Cheers.",

        "Customer: Can you confirm our renewal date?\n"
        "Agent: Your contract renews on the 14th of next month.\n"
        "Customer: Perfect, no changes needed from our side.",
    ],
    "quiet_drifter": [
        # Deliberately bland. The text gives away almost nothing.
        # This is the point: only the behavioural data flags this customer.
        "Customer: Password reset link isnt arriving.\n"
        "Agent: I've triggered a new one, please check spam.\n"
        "Customer: got it thanks",

        "Customer: How do I download an invoice?\n"
        "Agent: Billing > Invoices > Download PDF.\n"
        "Customer: ok",
    ],
    "loud_but_stable": [
        # Angry tone, but no churn intent, no competitor, no exit language.
        "Customer: This is really frustrating, the export keeps timing out again!\n"
        "Agent: I'm sorry about that, let me check the logs.\n"
        "Customer: Every single week something breaks. Please just fix it properly.\n"
        "Agent: Escalating to engineering now, I'll update you today.\n"
        "Customer: Fine. I do like the product, it just needs to be reliable.",

        "Customer: Why is the report layout changed AGAIN without warning??\n"
        "Agent: We shipped an update on Tuesday, I can share the release notes.\n"
        "Customer: You guys need better comms. Anyway send them over.",

        "Customer: The mobile app is so slow it's unusable today.\n"
        "Agent: There was a degradation this morning, now resolved.\n"
        "Customer: Ok its working now. Still annoying.",
    ],
    "billing_wounded": [
        "Customer: I've been charged twice this month. This is the second time!\n"
        "Agent: I apologise, I can see a duplicate charge on the 3rd.\n"
        "Customer: I want it refunded today, not in 7 working days.\n"
        "Agent: I'll request an expedited refund.\n"
        "Customer: If this happens again I'm escalating to my finance director.",

        "Customer: My card payment failed but I have funds. Now my account is limited.\n"
        "Agent: Let me re-run the payment for you.\n"
        "Customer: Third time I'm calling about this billing mess.",

        "Customer: Why did my invoice go up 22% with no notice?\n"
        "Agent: There was a plan price revision effective this quarter.\n"
        "Customer: Nobody told us. I need to review whether this still makes sense for us.",
    ],
    "escalating_churner": [
        "Customer: We've decided to cancel our subscription at the end of the term.\n"
        "Agent: I'm sorry to hear that, may I ask what's driving the decision?\n"
        "Customer: Four tickets in two months, none properly resolved. "
        "We're moving to a competitor next quarter.\n"
        "Agent: Can I arrange a call with your account manager?\n"
        "Customer: You can, but I doubt it changes anything at this point.",

        "Customer: I need to speak to a manager. This is the third time raising the same issue.\n"
        "Agent: I understand your frustration, let me escalate.\n"
        "Customer: We're already trialling an alternative platform. "
        "Honestly we're just waiting for our contract to end.",

        "Customer: please send me the offboarding process and data export options\n"
        "Agent: Before that, can we discuss what went wrong?\n"
        "Customer: Its too late for that. We've signed with another vendor.",

        "Customer: Still no fix. I'm done chasing this.\n"
        "Agent: I'm escalating to our senior team now.\n"
        "Customer: Save it. I'm taking this to our legal team if the refund isnt processed.",
    ],
    "new_fragile": [
        "Customer: I don't understand how to set up the integration, the docs are confusing.\n"
        "Agent: I can walk you through it, do you have 10 minutes?\n"
        "Customer: Not right now. Is there a video?",

        "Customer: We signed up last month but honestly we havent used it much yet.\n"
        "Agent: Would an onboarding session help?\n"
        "Customer: Maybe. My team hasnt had time to learn it.",

        "Customer: how do i import my existing data\n"
        "Agent: Settings > Import > choose CSV.\n"
        "Customer: tried that, got an error. ill try again later",
    ],
}


# --------------------------------------------------------------------------
# 3. HELPERS
# --------------------------------------------------------------------------

def pick(rng_tuple, integer=True):
    """Draw a random value from a (low, high) range. (None, None) -> None."""
    low, high = rng_tuple
    if low is None:
        return None
    if integer:
        return int(np.random.randint(low, high + 1))
    return round(float(np.random.uniform(low, high)), 2)


def assign_archetypes(n):
    """Build a list of n archetype names matching the target proportions."""
    names, weights = zip(*[(k, v["weight"]) for k, v in ARCHETYPES.items()])
    counts = [int(round(w * n)) for w in weights]
    # fix rounding drift so the total is exactly n
    while sum(counts) < n:
        counts[0] += 1
    while sum(counts) > n:
        counts[0] -= 1
    out = []
    for name, c in zip(names, counts):
        out.extend([name] * c)
    random.shuffle(out)
    return out


# --------------------------------------------------------------------------
# 4. CUSTOMER GENERATION
# --------------------------------------------------------------------------

def generate_customers(n=N_CUSTOMERS):
    archetype_list = assign_archetypes(n)
    rows = []

    for i, arch_name in enumerate(archetype_list, start=1):
        a = ARCHETYPES[arch_name]

        tenure = pick(a["tenure_months"])
        signup = TODAY - timedelta(days=tenure * 30 + random.randint(0, 28))
        segment = random.choice(SEGMENTS)

        # Revenue correlates with segment - realistic, and lets us compute
        # "revenue at risk" later in the dashboard.
        base_revenue = {"Enterprise": (4000, 20000), "Mid-Market": (1200, 5000),
                        "SMB": (300, 1500), "Startup": (60, 500)}[segment]
        revenue = round(random.uniform(*base_revenue), 2)

        tickets = pick(a["support_tickets_90d"])
        unresolved = int(round(tickets * random.uniform(*a["unresolved_frac"])))

        rows.append({
            "customer_id": f"CUST{i:04d}",
            "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "segment": segment,
            "plan_tier": random.choice(PLAN_TIERS),
            "monthly_revenue": revenue,
            "tenure_months": tenure,
            "signup_date": signup.strftime("%Y-%m-%d"),
            "last_login_days_ago": pick(a["last_login_days_ago"]),
            "logins_last_30d": pick(a["logins_last_30d"]),
            "usage_pct_change_30d": pick(a["usage_pct_change_30d"], integer=False),
            "support_tickets_90d": tickets,
            "unresolved_tickets_90d": unresolved,
            "avg_resolution_hours": pick(a["avg_resolution_hours"], integer=False),
            "failed_payments_90d": pick(a["failed_payments_90d"]),
            "days_since_last_payment": pick(a["days_since_last_payment"]),
            "nps_score": pick(a["nps_score"]),
            "csat_avg": pick(a["csat_avg"], integer=False),
            "plan_changed_last_90d": random.choice(a["plan_change"]),
            "contract_end_days": pick(a["contract_end_days"]),
            "region": random.choice(REGIONS),
            "churned_flag": int(random.random() < a["churn_prob"]),
            "_archetype": arch_name,   # kept for our own validation only
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. CONVERSATION GENERATION
# --------------------------------------------------------------------------

def generate_conversations(customers_df):
    rows = []
    conv_id = 1

    for _, cust in customers_df.iterrows():
        a = ARCHETYPES[cust["_archetype"]]
        n_conv = pick(a["n_conversations"])
        pool = TRANSCRIPTS[cust["_archetype"]]

        for _ in range(n_conv):
            # Recent conversations matter more, so skew timestamps toward now.
            days_ago = int(np.random.exponential(scale=25))
            days_ago = min(days_ago, 89)
            ts = TODAY - timedelta(days=days_ago,
                                   hours=random.randint(0, 23),
                                   minutes=random.randint(0, 59))

            transcript = random.choice(pool)
            channel = random.choice(CHANNELS)

            rows.append({
                "conversation_id": f"CONV{conv_id:05d}",
                "customer_id": cust["customer_id"],
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "channel": channel,
                "agent_id": f"AGT{random.randint(1, 12):03d}",
                "duration_minutes": round(random.uniform(2, 45), 1),
                "transcript": transcript,
                "resolved_flag": int(random.random() < (
                    0.9 if cust["_archetype"] in ("healthy_loyal", "quiet_drifter") else 0.45
                )),
            })
            conv_id += 1

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 6. DELIBERATE MESSINESS
# --------------------------------------------------------------------------
# Real data is dirty. We inject realistic problems so Phase 3 (cleaning) has
# genuine work to do - and so we can show a before/after data quality report.

def make_messy(df):
    df = df.copy()
    n = len(df)

    # a) inconsistent casing / whitespace in categorical columns
    idx = np.random.choice(n, size=int(n * 0.10), replace=False)
    df.loc[idx, "segment"] = df.loc[idx, "segment"].str.lower()
    idx = np.random.choice(n, size=int(n * 0.06), replace=False)
    df.loc[idx, "plan_tier"] = " " + df.loc[idx, "plan_tier"] + " "

    # b) extra missing values beyond the archetype-driven ones
    for col, frac in [("csat_avg", 0.08), ("avg_resolution_hours", 0.05),
                      ("region", 0.03)]:
        idx = np.random.choice(n, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # c) impossible values that range validation must catch
    idx = np.random.choice(n, size=3, replace=False)
    df.loc[idx, "tenure_months"] = -abs(df.loc[idx, "tenure_months"])   # negative tenure
    idx = np.random.choice(n, size=2, replace=False)
    df.loc[idx, "nps_score"] = 47                                       # out of 0-10 range
    idx = np.random.choice(n, size=2, replace=False)
    df.loc[idx, "csat_avg"] = 9.9                                       # out of 1-5 range

    # d) duplicate rows (same customer submitted twice)
    dupes = df.sample(3, random_state=SEED)
    df = pd.concat([df, dupes], ignore_index=True)

    return df



# --------------------------------------------------------------------------
# 7. DATA DICTIONARY
# --------------------------------------------------------------------------
# Generated alongside the data so it can never drift out of sync with the
# actual columns. Referenced from the README.

DICTIONARY = [
    ("customers.csv", "customer_id", "string", "Unique customer identifier", "Join key"),
    ("customers.csv", "name", "string", "Contact name", "Display only"),
    ("customers.csv", "segment", "category", "Enterprise / Mid-Market / SMB / Startup", "Grouping"),
    ("customers.csv", "plan_tier", "category", "Basic / Standard / Premium / Enterprise", "Context"),
    ("customers.csv", "monthly_revenue", "float", "Monthly recurring revenue", "Revenue-at-risk KPI"),
    ("customers.csv", "tenure_months", "int", "Months since signup", "NEW_CUSTOMER_FRAGILE"),
    ("customers.csv", "signup_date", "date", "Account creation date", "Cross-check on tenure"),
    ("customers.csv", "last_login_days_ago", "int", "Days since last login", "INACTIVITY"),
    ("customers.csv", "logins_last_30d", "int", "Login count, trailing 30 days", "LOGIN_DROP"),
    ("customers.csv", "usage_pct_change_30d", "float", "% change vs previous 30 days", "USAGE_DECLINE"),
    ("customers.csv", "support_tickets_90d", "int", "Tickets raised, trailing 90 days", "SUPPORT_SPIKE"),
    ("customers.csv", "unresolved_tickets_90d", "int", "Of those, still open", "UNRESOLVED_BACKLOG"),
    ("customers.csv", "avg_resolution_hours", "float", "Mean time to resolve", "SLOW_RESOLUTION"),
    ("customers.csv", "failed_payments_90d", "int", "Failed payment attempts", "PAYMENT_FAILURE"),
    ("customers.csv", "days_since_last_payment", "int", "Days since last successful payment", "OVERDUE"),
    ("customers.csv", "nps_score", "int, nullable", "Net Promoter Score 0-10; null = no response", "LOW_NPS / NPS_SILENCE"),
    ("customers.csv", "csat_avg", "float, nullable", "Mean satisfaction 1-5", "LOW_CSAT"),
    ("customers.csv", "plan_changed_last_90d", "category", "none / upgrade / downgrade", "DOWNGRADE"),
    ("customers.csv", "contract_end_days", "int", "Days until renewal date", "CONTRACT_ENDING"),
    ("customers.csv", "region", "category, nullable", "Geographic region", "Grouping"),
    ("conversations.csv", "conversation_id", "string", "Unique conversation identifier", "Join key"),
    ("conversations.csv", "customer_id", "string", "Owning customer", "Join key"),
    ("conversations.csv", "timestamp", "datetime", "When the conversation occurred", "Recency weighting"),
    ("conversations.csv", "channel", "category", "chat / email / phone_transcript", "Context"),
    ("conversations.csv", "agent_id", "string", "Support agent handling it", "Context"),
    ("conversations.csv", "duration_minutes", "float", "Conversation length", "Context"),
    ("conversations.csv", "transcript", "text", "Multi-turn customer/agent dialogue", "PRIMARY LLM INPUT"),
    ("conversations.csv", "resolved_flag", "int", "1 if the issue was resolved", "REPEAT_ISSUE support"),
    ("answer_key.csv", "customer_id", "string", "Join key", "Evaluation only"),
    ("answer_key.csv", "_archetype", "category", "Which recipe generated this customer", "Evaluation only"),
    ("answer_key.csv", "churned_flag", "int", "Ground truth churn label", "Evaluation only"),
]


def write_dictionary():
    pd.DataFrame(DICTIONARY, columns=["file", "column", "data_type", "meaning", "role"]) \
      .to_csv(os.path.join(OUT_DIR, "data_dictionary.csv"), index=False)


# --------------------------------------------------------------------------
# 8. MAIN
# --------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    customers = generate_customers()
    conversations = generate_conversations(customers)

    # The answer key stays in a separate file. The pipeline must never see it.
    answer_key = customers[["customer_id", "_archetype", "churned_flag"]].copy()
    answer_key.to_csv(os.path.join(OUT_DIR, "answer_key.csv"), index=False)

    # churned_flag lives ONLY in answer_key.csv. Keeping it in customers.csv
    # would let the pipeline read the answer - textbook target leakage.
    customers = customers.drop(columns=["_archetype", "churned_flag"])
    customers = make_messy(customers)

    customers.to_csv(os.path.join(OUT_DIR, "customers.csv"), index=False)
    conversations.to_csv(os.path.join(OUT_DIR, "conversations.csv"), index=False)
    write_dictionary()

    print(f"customers.csv      : {len(customers)} rows, {len(customers.columns)} columns")
    print(f"conversations.csv  : {len(conversations)} rows")
    print(f"answer_key.csv     : {len(answer_key)} rows (evaluation only)")
    print(f"data_dictionary.csv: {len(DICTIONARY)} column definitions\n")

    print("Archetype distribution:")
    print(answer_key["_archetype"].value_counts().to_string())
    print(f"\nChurn rate: {answer_key['churned_flag'].mean():.1%}")
    print(f"Customers with zero conversations: "
          f"{len(set(answer_key.customer_id) - set(conversations.customer_id))}")


if __name__ == "__main__":
    main()