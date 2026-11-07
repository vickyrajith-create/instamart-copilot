import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from lapse_data import get_split, make_preprocessor

feat, X_tr, X_te, y_tr, y_te = get_split()

# (b) Logistic regression: C in {0.01, 0.1, 1, 10, 100}
lr_pipe = Pipeline([("pre", make_preprocessor()), ("clf", LogisticRegression(max_iter=1000))])
lr_gs = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1")
lr_gs.fit(X_tr, y_tr)
print("LR best params:", lr_gs.best_params_, "| best CV F1:", round(lr_gs.best_score_, 4))

# (c) Random forest: max_depth x min_samples_leaf (45 fits)
rf_pipe = Pipeline([("pre", make_preprocessor()),
                    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_gs = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8],
                               "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1")
rf_gs.fit(X_tr, y_tr)
print("RF best params:", rf_gs.best_params_, "| best CV F1:", round(rf_gs.best_score_, 4))

# (d) Evaluate on the untouched test set, lapsed class only
for name, gs in [("Logistic regression", lr_gs), ("Random forest", rf_gs)]:
    pred = gs.best_estimator_.predict(X_te)
    print(f"\n{name} - test set")
    print("confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_te, pred))
    print("precision:", round(precision_score(y_te, pred, zero_division=0), 3),
          "| recall:", round(recall_score(y_te, pred, zero_division=0), 3),
          "| F1:", round(f1_score(y_te, pred, zero_division=0), 3))

joblib.dump({"lr": lr_gs.best_estimator_, "rf": rf_gs.best_estimator_,
             "lr_params": lr_gs.best_params_, "rf_params": rf_gs.best_params_},
            "tuned_models.joblib")
print("\nsaved tuned_models.joblib (for the next scripts)")