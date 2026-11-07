import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")
NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate",
       "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]


def build_dataset():
    orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv("data/clean/products.csv")
    orders = orders.merge(products[["product_id", "category"]], on="product_id")

    # features use ONLY orders dated before the cutoff
    before = orders[orders["order_ts"] < CUTOFF]
    done = before[before["status"] == "completed"]

    feat = done.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_delivery=("delivery_minutes", "mean")).reset_index()
    feat["recency"] = (CUTOFF - feat["last_order"]).dt.days

    n_all = before.groupby("customer_id")["order_id"].nunique()
    n_cancel = before[before["status"] == "cancelled"].groupby("customer_id")["order_id"].nunique()
    feat["cancel_rate"] = feat["customer_id"].map(n_cancel).fillna(0) / feat["customer_id"].map(n_all)

    cc = done.groupby(["customer_id", "category"]).size().reset_index(name="n")
    cc = cc.sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
    fav = cc.drop_duplicates("customer_id").set_index("customer_id")["category"]
    feat["fav_category"] = feat["customer_id"].map(fav)

    feat = feat.merge(customers[["customer_id", "city", "gender", "age", "signup_date"]],
                      on="customer_id")
    feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days

    # label: no completed order in September 2025 -> lapsed = 1
    sept = orders[(orders["status"] == "completed") &
                  (orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < END)]
    feat["lapsed"] = (~feat["customer_id"].isin(set(sept["customer_id"]))).astype(int)
    feat = feat.sort_values("customer_id").reset_index(drop=True)
    return feat, feat[NUM + CAT], feat["lapsed"]


def get_split():
    feat, X, y = build_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    return feat, X_tr, X_te, y_tr, y_te


def make_preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT),
        ("num", StandardScaler(), NUM)])


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, f1_score
    feat, X_tr, X_te, y_tr, y_te = get_split()
    print("cohort:", len(feat), "(reference 1156)")
    print("lapsed:", int(feat["lapsed"].sum()), f"({feat['lapsed'].mean()*100:.2f}%)  (reference 443 / 38.32%)")
    print("train/test:", len(X_tr), len(X_te), "(reference 924 / 232)")
    base = np.zeros(len(y_te), dtype=int)
    print("baseline accuracy:", round(accuracy_score(y_te, base) * 100, 1),
          "| lapsed F1:", f1_score(y_te, base, zero_division=0), "(reference 61.6 / 0)")

    pre = make_preprocessor().fit(X_tr)   # fitted on TRAIN rows only
    i = NUM.index("frequency")
    print("scaler mean of frequency (train only):", round(pre.named_transformers_["num"].mean_[i], 3))
    print("full-cohort mean of frequency:", round(feat["frequency"].mean(), 3))
    assert X_tr.isna().sum().sum() == 0 and X_te.isna().sum().sum() == 0
    print("no missing values in any feature column")