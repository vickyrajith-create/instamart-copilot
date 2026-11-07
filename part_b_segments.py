import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

rfm = pd.read_csv("rfm_table.csv")
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])

K = 4
rfm["segment_id"] = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(X)

prof = rfm.groupby("segment_id").agg(
    size=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")).round(2)

# Name segments by their profile (cluster numbers are only labels, with no ordering)
names, left = {}, set(prof.index)
premium = prof.loc[list(left), "mean_monetary"].idxmax()
names[premium] = "Premium High Spenders"; left.discard(premium)
loyal = prof.loc[list(left), "mean_frequency"].idxmax()
names[loyal] = "Loyal Regulars"; left.discard(loyal)
lapsing = prof.loc[list(left), "mean_recency"].idxmax()
names[lapsing] = "Lapsing Customers"; left.discard(lapsing)
for s in left:
    names[s] = "Occasional Low-Value"

prof["segment_name"] = prof.index.map(names)
print(prof.to_string())

rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium
os.makedirs("outputs", exist_ok=True)
rfm[["customer_id", "recency", "frequency", "monetary",
     "segment_id", "segment_name", "is_premium"]].to_csv("outputs/customer_segments.csv", index=False)

pca = PCA(n_components=2).fit(X)
r = pca.explained_variance_ratio_
print("\nPCA explained variance:", r.round(3), "| total:", round(r.sum(), 3), "(reference 0.634 / 0.218 / 0.852)")
print("premium rows:", int(rfm["is_premium"].sum()))