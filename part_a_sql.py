import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
raw = open("sql/queries.sql", encoding="utf-8").read()
text = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("--"))
queries = [q.strip() for q in text.split(";") if q.strip()]
assert len(queries) == 7, f"expected 7 queries, found {len(queries)}"

res = {}
for i, q in enumerate(queries, 1):
    res[i] = pd.read_sql_query(q, con)
con.close()

print("Q1 customers:", len(res[1]), "(reference 30)")
print("Q2 rows:", len(res[2]), "| total:", round(res[2]["revenue"].sum(), 2),
      "| largest:", res[2].iloc[0].tolist(), "(reference 30 / 2998668.80 / Bengaluru Electronics 233909.45)")
print("Q3 products:", len(res[3]), "(reference 11)")
print("Q4 customers:", len(res[4]), "(reference 25)")
print("Q5 customers:", len(res[5]), "| average:", res[5]["avg_total"].iloc[0], "(reference 550 / 2552.06)")
q6 = res[6]
print("Q6 rows where RANK != ROW_NUMBER:", int((q6["rnk"] != q6["row_num"]).sum()), "(reference 8)")
print("Q6 top customer per city:")
print(q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q7 days:", len(res[7]), "| last cumulative:", res[7]["cumulative_revenue"].iloc[-1],
      "(reference 183 / 2998668.80)")