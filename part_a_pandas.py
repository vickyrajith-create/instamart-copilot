import sqlite3
import numpy as np
import pandas as pd

con = sqlite3.connect("data/instamart.db")
orders = pd.read_sql_query("SELECT * FROM orders", con)
customers = pd.read_sql_query("SELECT * FROM customers", con)
products = pd.read_sql_query("SELECT * FROM products", con)
con.close()

done = orders[orders["status"] == "completed"]
m = done.merge(customers[["customer_id", "city"]], on="customer_id").merge(
    products[["product_id", "category"]], on="product_id")

# (a) pivot table with margins -> grand total must equal the SQL total
pv = pd.pivot_table(m, values="order_amount", index="city", columns="category",
                    aggfunc="sum", margins=True, margins_name="Total")
print("Pivot grand total:", round(pv.loc["Total", "Total"], 2), "(SQL total 2998668.80)")
print(pv.round(2).to_string())

by_city = m.groupby("city").agg(total=("order_amount", "sum"),
                                mean=("order_amount", "mean"),
                                count=("order_amount", "count")).round(2)
print("\nPer city:\n", by_city)

# (b) distribution statistics for completed order_amount
x = done["order_amount"]
q1, q2, q3 = x.quantile([0.25, 0.5, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("\nmean", round(x.mean(), 2), "| median", round(q2, 2), "| skew", round(x.skew(), 2))
print("Q1", round(q1, 2), "| Q3", round(q3, 2), "| IQR", round(iqr, 2))
print("lower fence", round(lo, 2), "| upper fence", round(hi, 2))
print("outliers below:", int((x < lo).sum()), "| above:", int((x > hi).sum()))

cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr = done[cols].corr()
print("\nCorrelation matrix:\n", corr.round(3))
pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
print("Largest |corr| pair:", pairs.abs().idxmax(), round(pairs[pairs.abs().idxmax()], 3))