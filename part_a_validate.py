import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
cust = pd.read_sql_query("SELECT * FROM customers", con)
ords = pd.read_sql_query("SELECT * FROM orders", con)
con.close()

done = ords[ords["status"] == "completed"]
cancelled = ords[ords["status"] == "cancelled"]

checks = [
    ("six cities", cust["city"].nunique() == 6),
    ("city counts", cust["city"].value_counts().to_dict() ==
        {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}),
    ("gender counts", cust["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}),
    ("no missing ages", cust["age"].isna().sum() == 0),
    ("ages between 18 and 60", cust["age"].between(18, 60).all()),
    ("12,309 order rows", len(ords) == 12309),
    ("one row per order_id", ords["order_id"].is_unique),
    ("11,110 completed", len(done) == 11110),
    ("1,199 cancelled", len(cancelled) == 1199),
    ("no pending rows", (ords["status"] == "pending").sum() == 0),
    ("0 missing order_amount", ords["order_amount"].isna().sum() == 0),
    ("0 missing delivery (completed)", done["delivery_minutes"].isna().sum() == 0),
    ("1,199 missing delivery (cancelled)", cancelled["delivery_minutes"].isna().sum() == 1199),
    ("completed revenue = 2,998,668.80", round(done["order_amount"].sum(), 2) == 2998668.80),
]
for name, ok in checks:
    assert ok, f"CHECK FAILED: {name}"
print("ALL CHECKPOINTS PASSED")