import re
import pandas as pd
from mock_llm import est_tokens

MAX_INPUT_TOKENS = 400
PATTERNS = [  # applied in this exact order
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]


def redact(text):
    counts = {}
    for name, pat, placeholder in PATTERNS:
        text, n = pat.subn(placeholder, text)
        counts[name] = n
    return text, counts


def load_tickets():
    return pd.read_csv("data/raw/tickets.csv", keep_default_na=False)


if __name__ == "__main__":
    t = load_tickets()
    print("tickets:", len(t))

    empty = t["text"].str.strip() == ""
    tokens = t["text"].apply(est_tokens)
    long_ = tokens > MAX_INPUT_TOKENS
    print("empty tickets:", int(empty.sum()), "(reference 4)")
    print("long tickets:", int(long_.sum()), "(reference 4) | est tokens of long ones:",
          sorted(set(tokens[long_])), "(reference 734)")

    totals = {"email": 0, "phone": 0, "order_id": 0}
    redacted = []
    for text in t["text"]:
        r, c = redact(text)
        redacted.append(r)
        for k in totals:
            totals[k] += c[k]
    print("substitutions:", totals, "| total:", sum(totals.values()),
          "(reference 55 / 91 / 132 / 278)")

    left = sum(len(p.findall(r)) for r in redacted for _, p, _ in PATTERNS)
    print("patterns found in redacted texts:", left, "(reference 0)")
    assert left == 0