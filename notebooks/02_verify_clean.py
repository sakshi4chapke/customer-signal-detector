import pandas as pd

c = pd.read_csv("data/processed/customers_clean.csv")
v = pd.read_csv("data/processed/conversations_clean.csv")

print("SHAPE:", c.shape, v.shape)

print("\nNo duplicate IDs:", c.customer_id.is_unique)
print("Segments:", sorted(c.segment.unique()))
print("Plan tiers:", sorted(c.plan_tier.unique()))
print("NPS range:", c.nps_score.min(), "-", c.nps_score.max())
print("CSAT range:", c.csat_avg.min(), "-", c.csat_avg.max())
print("Min tenure:", c.tenure_months.min())

print("\nNEW COLUMNS")
print(c[["customer_id", "nps_score", "nps_score_missing",
         "unresolved_ratio", "annual_value"]].head())

print("\nREMAINING NULLS (should only be nps_score / csat_avg)")
print(c.isnull().sum()[lambda s: s > 0])

print("\nRECENCY (days_ago) — should skew toward recent")
print(v.days_ago.describe().round(1))