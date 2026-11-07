import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

rfm = pd.read_csv("rfm_table.csv")
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])

rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"K": k, "inertia": round(km.inertia_, 1),
                 "silhouette": round(silhouette_score(X, km.labels_), 3)})
table = pd.DataFrame(rows)
print(table.to_string(index=False))