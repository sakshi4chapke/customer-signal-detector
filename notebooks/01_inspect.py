import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

customers = pd.read_csv("data/raw/customers.csv")
conversations = pd.read_csv("data/raw/conversations.csv")
key = pd.read_csv("data/raw/answer_key.csv")

print("SHAPE")
print("customers    ", customers.shape)
print("conversations", conversations.shape)

print("\nFIRST 5 CUSTOMERS")
print(customers.head())

print("\nTYPES AND NON-NULL COUNTS")
print(customers.info())

print("\nNUMERIC SUMMARY")
print(customers.describe().round(1))

print("\nMISSING VALUES")
print(customers.isnull().sum()[lambda s: s > 0])

print("\nDUPLICATE CUSTOMER IDS")
print(customers.duplicated(subset=["customer_id"]).sum())

print("\nCATEGORY SPELLING CHECK")
print(customers["segment"].value_counts())

print("\nARCHETYPE SEPARATION")
df = customers.drop_duplicates("customer_id").merge(key, on="customer_id")
cols = ["last_login_days_ago", "usage_pct_change_30d",
        "support_tickets_90d", "failed_payments_90d", "churned_flag"]
print(df.groupby("_archetype")[cols].mean().round(1))

print("\nSAMPLE TRANSCRIPT — quiet drifter")
qd = df[df._archetype == "quiet_drifter"].customer_id.head(1).item()
print(conversations[conversations.customer_id == qd].transcript.head(1).item())

print("\nSAMPLE TRANSCRIPT — escalating churner")
ec = df[df._archetype == "escalating_churner"].customer_id.head(1).item()
print(conversations[conversations.customer_id == ec].transcript.head(1).item())