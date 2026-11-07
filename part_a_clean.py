import pandas as pd

kw = dict(keep_default_na=False, na_values=[""])
customers = pd.read_csv("data/raw/customers.csv", **kw)

# (a) City: strip spaces + title case -> exactly six cities
customers["city"] = customers["city"].str.strip().str.title()

# (b) Gender: every spelling -> male / female / unknown (blank -> unknown)
first = customers["gender"].fillna("").str.strip().str.lower().str[:1]
customers["gender"] = first.map({"m": "male", "f": "female"}).fillna("unknown")

# (c) Age: values outside 10-100 become missing, then fill with the city median
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
bad_age = (customers["age"] < 10) | (customers["age"] > 100)
print("ages set to missing (outside 10-100):", int(bad_age.sum()))
customers.loc[bad_age, "age"] = None
n_missing = int(customers["age"].isna().sum())
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)
print("ages filled with city median:", n_missing)

# (d) signup_date -> datetime
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# Checks against the reference values
print(customers["city"].value_counts())
print(customers["gender"].value_counts())
print("missing ages:", int(customers["age"].isna().sum()))
print("age min/max:", customers["age"].min(), customers["age"].max())
# ======================= ORDERS =======================
print("\n===== CLEAN ORDERS =====")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv", **kw)
print("raw rows:", len(orders))

# Rule 1: order_ts -> datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Rule 2: remove exact duplicate rows
before = len(orders)
orders = orders.drop_duplicates()
print("Rule 2 - exact duplicates removed:", before - len(orders))

# Rule 3: one record per order_id = the one with the latest order_ts
before = len(orders)
orders = (orders.sort_values("order_ts")
                .drop_duplicates(subset="order_id", keep="last")
                .sort_values("order_id")
                .reset_index(drop=True))
print("Rule 3 - older records removed (pending):", before - len(orders))

# Rule 4: order_amount -> numeric; fill missing from product price
price = orders["product_id"].map(products.set_index("product_id")["price_inr"])
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
n_bad = int(orders["order_amount"].isna().sum())
calc = (price * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
orders["order_amount"] = orders["order_amount"].fillna(calc)
print("Rule 4 - amounts filled from product price:", n_bad)

# Rule 5: delivery_minutes
orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
neg = orders["delivery_minutes"] < 0
print("Rule 5a - negative delivery set to missing:", int(neg.sum()))
orders.loc[neg, "delivery_minutes"] = None

orders["category"] = orders["product_id"].map(products.set_index("product_id")["category"])
done = orders["status"] == "completed"
cat_median = orders[done].groupby("category")["delivery_minutes"].transform("median")
n_fill = int(orders.loc[done, "delivery_minutes"].isna().sum())
orders.loc[done, "delivery_minutes"] = orders.loc[done, "delivery_minutes"].fillna(cat_median)
print("Rule 5b - completed delivery filled with category median:", n_fill)
orders = orders.drop(columns="category")   # helper column, not part of the clean table

# Checks against the reference values
print("\nrows:", len(orders), "(reference 12309)")
print(orders["status"].value_counts().to_dict(), "(reference completed 11110, cancelled 1199)")
print("missing order_amount:", int(orders["order_amount"].isna().sum()), "(reference 0)")
print("missing delivery, completed:", int(orders.loc[orders["status"] == "completed", "delivery_minutes"].isna().sum()), "(reference 0)")
print("missing delivery, cancelled:", int(orders.loc[orders["status"] == "cancelled", "delivery_minutes"].isna().sum()), "(reference 1199)")
print("sum amount, completed:", round(orders.loc[orders["status"] == "completed", "order_amount"].sum(), 2), "(reference 2998668.80)") 
# ======================= STORE =======================
import os, sqlite3

os.makedirs("data/clean", exist_ok=True)

# store dates as text in the form the brief asks for
customers_out = customers.copy()
customers_out["signup_date"] = customers_out["signup_date"].dt.strftime("%Y-%m-%d")
orders_out = orders.copy()
orders_out["order_ts"] = orders_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

customers_out.to_csv("data/clean/customers.csv", index=False)
products.to_csv("data/clean/products.csv", index=False)
orders_out.to_csv("data/clean/orders.csv", index=False)

if os.path.exists("data/instamart.db"):
    os.remove("data/instamart.db")
con = sqlite3.connect("data/instamart.db")
customers_out.to_sql("customers", con, index=False)
products.to_sql("products", con, index=False)
orders_out.to_sql("orders", con, index=False)
con.close()
print("\nStored: data/clean/*.csv and data/instamart.db")