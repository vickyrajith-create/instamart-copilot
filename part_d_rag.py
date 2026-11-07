import os
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

POLICY_DIR = os.path.join("data", "policies")
SYSTEM_PROMPT = ("You are a POLICY ASSISTANT. Answer only from the supplied CONTEXT. "
                 "If the answer is not in the context, say that you could not find the information.")


def chunk_words(text, size=40, overlap=10):
    words, step, chunks, start = text.split(), size - overlap, [], 0
    while True:
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):          # stop after the chunk that reaches the last word
            break
        start += step
    return chunks


def build_index(overlap=10, size=40):
    chunks = []
    for fn in sorted(os.listdir(POLICY_DIR)):
        if fn.endswith(".txt"):
            with open(os.path.join(POLICY_DIR, fn), encoding="utf-8") as f:
                text = f.read()
            for c in chunk_words(text, size, overlap):
                chunks.append({"source": fn, "text": c})
    vec = TfidfVectorizer()
    mat = vec.fit_transform([c["text"] for c in chunks])
    return {"chunks": chunks, "vec": vec, "mat": mat}


INDEX = build_index(overlap=10)


def retrieve(query, k=3, index=None):
    idx = index if index is not None else INDEX
    sims = cosine_similarity(idx["vec"].transform([query]), idx["mat"])[0]
    order = sims.argsort()[::-1][:k]
    return [{"source": idx["chunks"][i]["source"], "score": float(sims[i]),
             "text": idx["chunks"][i]["text"]} for i in order]


def answer(question, index=None):
    hits = retrieve(question, 3, index)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
    resp = mock_llm.chat(messages)
    return resp.choices[0].message.content, [h["source"] for h in hits]


def evaluate(index, label):
    qa = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    n_ans = doc_ok = fact_ok = n_un = fb_ok = 0
    failures = []
    for _, r in qa.iterrows():
        text, sources = answer(r["question"], index)
        if r["expected_doc"] == "NONE":
            n_un += 1
            if text.startswith("I could not find"):
                fb_ok += 1
            else:
                failures.append((r["question"], "should have said it could not find the answer"))
        else:
            n_ans += 1
            d, f = r["expected_doc"] in sources, r["expected_fact"] in text
            doc_ok += d
            fact_ok += f
            if not (d and f):
                failures.append((r["question"], f"doc found={d}, fact found={f} | answer: {text}"))
    print(f"\n=== {label} ===")
    print(f"expected_doc in top 3: {doc_ok}/{n_ans} | answer contains expected_fact: {fact_ok}/{n_ans} "
          f"| fallback on unanswerable: {fb_ok}/{n_un}  (reference 12/12, 12/12, 2/2)")
    for q, why in failures:
        print("  FAILS:", q, "->", why)


if __name__ == "__main__":
    counts = Counter(c["source"] for c in INDEX["chunks"])
    print("chunks:", len(INDEX["chunks"]), dict(counts), "(reference 15: refund 4, delivery 3, escalation 3, payment 3, privacy 2)")
    top = retrieve("refund policy late delivery")
    print("top 3 sources for 'refund policy late delivery':", [h["source"] for h in top], "(reference: all refund_policy.txt)")
    evaluate(INDEX, "OVERLAP 10")
    evaluate(build_index(overlap=0), "OVERLAP 0")