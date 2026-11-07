import json, logging, os, random, re, time
import pandas as pd
import mock_llm
from mock_llm import chat, est_tokens, TransientLLMError
from part_c_redact import PATTERNS, redact, load_tickets, MAX_INPUT_TOKENS

ALLOWED = ["billing", "delivery", "technical", "fraud"]
BASE_DELAY, MAX_RETRIES = 0.05, 3

ZERO_SHOT = ("You CLASSIFY customer support tickets into exactly one category. "
             "Allowed categories: billing, delivery, technical, fraud. "
             'Reply with JSON only, in the form {"category": "...", "reason": "..."}.')

FEW_SHOT = (ZERO_SHOT + "\nEXAMPLES:\n"
            'Ticket: I was charged twice for order [ORDER_ID]. Answer: {"category": "billing", "reason": "double charge"}\n'
            'Ticket: My order has not arrived and the rider is not picking up. Answer: {"category": "delivery", "reason": "missing order"}\n'
            'Ticket: The app freezes when I open the cart. Answer: {"category": "technical", "reason": "app problem"}\n'
            'Ticket: There is an unauthorized transaction on my card. Answer: {"category": "fraud", "reason": "unauthorized payment"}')

os.makedirs("outputs", exist_ok=True)
logging.basicConfig(filename="outputs/retry_log.txt", filemode="w", level=logging.INFO, format="%(message)s")
STATS = {"transient": 0}
WAITS = []


def call_with_retry(messages, ticket_id, label):
    retries = 0
    while True:
        try:
            return chat(messages)
        except TransientLLMError:
            STATS["transient"] += 1
            if retries >= MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** retries * random.uniform(0.9, 1.15)
            WAITS.append((retries, wait))
            logging.info(f"[{label}] ticket={ticket_id} retry={retries} wait={wait:.4f}s")
            time.sleep(wait)
            retries += 1


def validate(content):
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (content or "").lower()))
        found = [c for c in ALLOWED if c in words]
        return ("classified_keyword", found[0]) if len(found) == 1 else ("flagged_invalid_output", "")
    cat = data.get("category") if isinstance(data, dict) else None
    if isinstance(cat, str) and cat in ALLOWED:
        return "classified_json", cat
    return "flagged_invalid_output", ""


def run_pipeline(label, system_prompt, budget=None):
    mock_llm.reset()                         # reset immediately before each full run
    STATS["transient"] = 0
    total, rows = 0, []
    for _, t in load_tickets().iterrows():
        text, status, cat, used = t["text"], None, "", 0
        if text.strip() == "":
            status = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        elif budget is not None and total >= budget:
            status = "budget_exhausted"
        else:
            red, _ = redact(text)
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": red}]
            for m in messages:               # no email / phone / order id may reach the model
                assert not any(p.search(m["content"]) for _, p, _ in PATTERNS)
            try:
                resp = call_with_retry(messages, t["ticket_id"], label)
            except TransientLLMError:
                status = "flagged_error"
            else:
                used = resp.usage.total_tokens
                total += used
                status, cat = validate(resp.choices[0].message.content)
        rows.append({"ticket_id": t["ticket_id"], "customer_id": t["customer_id"], "status": status,
                     "category": cat, "tokens_used": used, "true_category": t["true_category"]})
    return pd.DataFrame(rows), total, STATS["transient"]


HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted", "flagged_error"}
seg = pd.read_csv("outputs/customer_segments.csv")
SEG_NAME = dict(zip(seg["customer_id"], seg["segment_name"]))
PREMIUM = dict(zip(seg["customer_id"], seg["is_premium"]))


def route_row(r):
    if r["status"] in HUMAN:
        return "human_review", "NORMAL"
    if r["category"] == "fraud":
        return "security_team", "HIGH"
    if PREMIUM.get(r["customer_id"], False):
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


def save_output(df, path):
    out = df.copy()
    out["segment_name"] = out["customer_id"].map(SEG_NAME).fillna("")
    rt = out.apply(route_row, axis=1)
    out["route"] = [r[0] for r in rt]
    out["priority"] = [r[1] for r in rt]
    out = out[["ticket_id", "customer_id", "status", "category", "segment_name",
               "priority", "route", "tokens_used"]]
    out.to_csv(path, index=False)
    return out


def summarize(name, df, total, transient):
    ok = df[df["status"].isin(["classified_json", "classified_keyword"])]
    correct = int((ok["category"] == ok["true_category"]).sum())
    print(f"\n=== {name} ===")
    print("status counts:", df["status"].value_counts().to_dict())
    print("transient errors:", transient, "(reference 19, all recovered)")
    print(f"classified: {len(ok)} | correct: {correct} | accuracy: {correct / len(ok) * 100:.1f}%")
    print(f"total tokens: {total} | tokens per classified ticket: {total / len(ok):.1f}")
    return ok


zero, z_tot, z_err = run_pipeline("zero-shot", ZERO_SHOT)
few, f_tot, f_err = run_pipeline("few-shot", FEW_SHOT)
ok = summarize("ZERO-SHOT (no budget)", zero, z_tot, z_err)
summarize("FEW-SHOT (no budget)", few, f_tot, f_err)

print("\nZero-shot confusion table (rows = true, columns = predicted):")
print(pd.crosstab(ok["true_category"], ok["category"]))
print("prompt tokens: zero-shot", est_tokens(ZERO_SHOT), "| few-shot", est_tokens(FEW_SHOT))

waits_ok = all(BASE_DELAY * 2 ** n * 0.9 <= w <= BASE_DELAY * 2 ** n * 1.15 for n, w in WAITS)
print("\nretry log lines:", len(WAITS), "| waits within jitter range:", waits_ok,
      "| min/max wait:", round(min(w for _, w in WAITS), 4), round(max(w for _, w in WAITS), 4))

# (a) budget demo: zero-shot with TOKEN_BUDGET = 6000
demo, d_tot, _ = run_pipeline("budget-demo", ZERO_SHOT, budget=6000)
save_output(demo, "outputs/triage_budget_demo.csv")
print("\nBUDGET DEMO (6000): budget_exhausted =", int((demo["status"] == "budget_exhausted").sum()),
      "| tokens used:", d_tot)

# (b) final results: few-shot with TOKEN_BUDGET = 100000
final, fin_tot, _ = run_pipeline("final", FEW_SHOT, budget=100000)
res = save_output(final, "outputs/triage_results.csv")
print("\nFINAL: rows", len(res), "(reference 164) | routes:", res["route"].value_counts().to_dict())
print("priorities:", res["priority"].value_counts().to_dict())