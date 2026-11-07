import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from lapse_data import get_split, make_preprocessor

feat, X_tr, X_te, y_tr, y_te = get_split()
saved = joblib.load("tuned_models.joblib")
C = saved["lr_params"]["clf__C"]

# unbalanced model, as tuned
base_pred = saved["lr"].predict(X_te)

# oversample the TRAINING rows only: keep every row, add random duplicates of lapsed rows
n0, n1 = int((y_tr == 0).sum()), int((y_tr == 1).sum())
extra = X_tr[y_tr == 1].sample(n=n0 - n1, replace=True, random_state=42)
X_bal = pd.concat([X_tr, extra])
y_bal = pd.concat([y_tr, pd.Series(1, index=extra.index)])
print("train class counts before:", n0, n1, "| after:", int((y_bal == 0).sum()), int((y_bal == 1).sum()))

bal = Pipeline([("pre", make_preprocessor()),
                ("clf", LogisticRegression(max_iter=1000, C=C))]).fit(X_bal, y_bal)
bal_pred = bal.predict(X_te)   # evaluated on the original, unchanged test set

def show(name, p):
    print(f"{name}: precision {precision_score(y_te, p, zero_division=0):.3f} | "
          f"recall {recall_score(y_te, p, zero_division=0):.3f} | "
          f"F1 {f1_score(y_te, p, zero_division=0):.3f}")

show("Logistic regression, un-balanced", base_pred)
show("Logistic regression, over-sampled", bal_pred)