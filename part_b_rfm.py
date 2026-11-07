import pandas as pd

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
done = orders[orders["status"] == "completed"]

snapshot = pd.Timestamp("2025-10-01 00:00:00")
rfm = done.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
).reset_index()
rfm["recency"] = (snapshot - rfm["last_order"]).dt.days
rfm["monetary"] = rfm["monetary"].round(2)
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

print("customers:", len(rfm), "(reference 1175)")
print("recency median/min/max:", rfm["recency"].median(), rfm["recency"].min(), rfm["recency"].max(), "(reference 19 / 0 / 180)")
print("frequency mean/max:", round(rfm["frequency"].mean(), 2), rfm["frequency"].max(), "(reference 9.46 / 33)")
print("monetary mean/max:", round(rfm["monetary"].mean(), 2), rfm["monetary"].max(), "(reference 2552.06 / 14766.55)")

rfm.to_csv("rfm_table.csv", index=False)