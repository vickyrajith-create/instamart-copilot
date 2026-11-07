import joblib
import numpy as np
import pandas as pd
from lapse_data import get_split, NUM, CAT

feat, X_tr, X_te, y_tr, y_te = get_split()
saved = joblib.load("tuned_models.joblib")
rf, lr = saved["rf"], saved["lr"]

def feature_names(pipe):
    pre = pipe.named_steps["pre"]
    cat = pre.named_transformers_["cat"].get_feature_names_out(CAT)
    return list(cat) + NUM

# (a) five most important features of the random forest
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=feature_names(rf))
print("Top 5 random forest features:\n", imp.sort_values(ascending=False).head(5).round(4))

# signs of three coefficients in the logistic regression
coef = pd.Series(lr.named_steps["clf"].coef_[0], index=feature_names(lr))
print("\nLogistic regression coefficients (standardised features):")
for f in ["recency", "frequency", "monetary"]:
    print(f"  {f}: {coef[f]:+.3f}")

# (b) save the selected model (random forest), reload and compare predictions
joblib.dump(rf, "outputs/lapse_model.joblib")
rf2 = joblib.load("outputs/lapse_model.joblib")
same = np.array_equal(rf.predict(X_te), rf2.predict(X_te))
print("\nreloaded model reproduces test predictions exactly:", same)
assert same

# (c) retention list: Premium customers in the test set, ranked by lapse probability
seg = pd.read_csv("outputs/customer_segments.csv")
premium = set(seg.loc[seg["is_premium"], "customer_id"])
test = feat.loc[X_te.index, ["customer_id", "lapsed"]].copy()
test["lapse_probability"] = rf.predict_proba(X_te)[:, 1]
prem = test[test["customer_id"].isin(premium)].sort_values("lapse_probability", ascending=False)
top = prem.head(10).rename(columns={"lapsed": "actually_lapsed"})
top["lapse_probability"] = top["lapse_probability"].round(4)
top[["customer_id", "lapse_probability", "actually_lapsed"]].to_csv("outputs/retention_list.csv", index=False)
print("\nPremium customers in the test set:", len(prem))
print(top[["customer_id", "lapse_probability", "actually_lapsed"]].to_string(index=False))
print("truly lapsed among the list:", int(top["actually_lapsed"].sum()), "of", len(top))