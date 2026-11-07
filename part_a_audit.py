import pandas as pd

# keep_default_na=False means only truly blank cells become missing;
# text like "N/A" stays as text so we can count it as a bad amount.
kw = dict(keep_default_na=False, na_values=[""])
customers = pd.read_csv("data/raw/customers.csv", **kw)
products = pd.read_csv("data/raw/products.csv", **kw)
orders = pd.read_csv("data/raw/orders.csv", **kw)

for name, df in [("customers", customers), ("products", products), ("orders", orders)]:
    print(f"\n===== {name} =====")
    print("shape:", df.shape)
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
    })
    print(summary)

amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
delivery_num = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
age_num = pd.to_numeric(customers["age"], errors="coerce")

audit = {
    "raw orders rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "pending rows": int((orders["status"] == "pending").sum()),
    "distinct order_id": int(orders["order_id"].nunique()),
    "non-numeric order_amount": int(amount_num.isna().sum()),
    "negative delivery_minutes": int((delivery_num < 0).sum()),
    "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
    "customers: blank age": int(customers["age"].isna().sum()),
    "customers: age outside 10-100": int(((age_num < 10) | (age_num > 100)).sum()),
    "customers: distinct raw city spellings": int(customers["city"].nunique()),
    "customers: distinct non-blank gender spellings": int(customers["gender"].nunique()),
    "customers: blank gender": int(customers["gender"].isna().sum()),
}
print("\n===== AUDIT FIGURES =====")
for k, v in audit.items():
    print(f"{k}: {v}")