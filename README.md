# Instamart Customer Intelligence & Support Copilot

## Overview

This repository contains the complete Customer Intelligence & Support Copilot capstone, built in four connected Parts on synthetic, fictional data. All monetary values are in Indian Rupees (INR). No network call, API key or paid service is used anywhere: every language-model call goes through the supplied offline file `mock_llm.py`.

| Part | What it does | Main files |
|---|---|---|
| A | Audits and cleans the raw exports, stores them in SQLite, answers business questions in SQL and pandas | `part_a_*.py`, `sql/queries.sql`, `data/instamart.db` |
| B | RFM segments (K-means) and a lapse-prediction model | `part_b_*.py`, `lapse_data.py`, `outputs/` |
| C | Support-ticket triage with privacy, retry, validation, budget and routing guardrails | `part_c_*.py` |
| D | Policy RAG assistant and a resumable refund agent | `part_d_*.py` |

`generate_data.py` and `mock_llm.py` are the supplied files and are committed unmodified. Random seeds are fixed wherever randomness appears (42).

## Reproduce

Run these commands from the repository root on Windows PowerShell (Python 3.11 or newer; the project was run on Python 3.14.5). On Mac or Linux, activate with `source .venv/bin/activate`.

1. `python -m venv .venv`
2. `.\.venv\Scripts\Activate.ps1`
3. `python -m pip install -r requirements.txt`
4. `python generate_data.py` (last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
5. `python part_a_audit.py`
6. `python part_a_clean.py` (writes `data/clean/*.csv` and `data/instamart.db`)
7. `python part_a_validate.py` (prints `ALL CHECKPOINTS PASSED`)
8. `python part_a_sql.py`
9. `python part_a_pandas.py`
10. `python part_b_rfm.py`
11. `python part_b_kmeans.py`
12. `python part_b_segments.py` (writes `outputs/customer_segments.csv`)
13. `python lapse_data.py`
14. `python part_b_models.py` (writes `tuned_models.joblib`)
15. `python part_b_balance.py`
16. `python part_b_final.py` (writes `outputs/lapse_model.joblib` and `outputs/retention_list.csv`)
17. `python part_c_redact.py`
18. `python part_c_triage.py` (writes `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv`, `outputs/retry_log.txt`)
19. `python part_d_rag.py`
20. `python part_d_run.py` (writes `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl` and the `state/` checkpoints)

Libraries: Python standard library, pandas, numpy, scikit-learn, joblib.

## Part A

### Audit of the raw data

| Table | Shape |
|---|---|
| customers.csv | 1,200 x 5 |
| products.csv | 40 x 4 |
| orders.csv | 13,084 x 9 |

**customers.csv**

| Column | dtype | Missing | Missing % |
|---|---|---|---|
| customer_id | str | 0 | 0.00 |
| city | str | 0 | 0.00 |
| signup_date | str | 0 | 0.00 |
| age | float64 | 61 | 5.08 |
| gender | str | 30 | 2.50 |

**products.csv**

| Column | dtype | Missing | Missing % |
|---|---|---|---|
| product_id | str | 0 | 0.0 |
| product_name | str | 0 | 0.0 |
| category | str | 0 | 0.0 |
| price_inr | int64 | 0 | 0.0 |

**orders.csv**

| Column | dtype | Missing | Missing % |
|---|---|---|---|
| order_id | str | 0 | 0.00 |
| customer_id | str | 0 | 0.00 |
| product_id | str | 0 | 0.00 |
| quantity | int64 | 0 | 0.00 |
| order_ts | str | 0 | 0.00 |
| status | str | 0 | 0.00 |
| discount_pct | int64 | 0 | 0.00 |
| order_amount | str | 0 | 0.00 |
| delivery_minutes | float64 | 2,541 | 19.42 |

`order_amount` is read as text because some entries are the words `N/A` or `error`.

**Audit figures** (all match the reference values)

| Figure | Value |
|---|---|
| Raw orders rows | 13,084 |
| Exact duplicate rows | 119 |
| Rows with status pending | 656 |
| Distinct order_id | 12,309 |
| order_amount entries that cannot be converted to a number | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| customers: blank age | 61 |
| customers: age outside 10-100 | 6 |
| customers: distinct raw city spellings | 24 |
| customers: distinct non-blank raw gender spellings | 10 |
| customers: blank gender | 30 |

### Cleaning rules

**Customers**

| Rule | Values changed |
|---|---|
| City: strip spaces and apply title case, leaving exactly six cities | all 1,200 rows standardised |
| Gender: every spelling mapped to male, female or unknown (blank becomes unknown) | 1,200 rows standardised, 30 blanks became unknown |
| Age outside 10-100 set to missing | 6 |
| Missing ages filled with the median age of the customer's city (groupby and transform) | 67 (61 blank plus the 6 above) |
| signup_date converted to datetime | 1,200 |

After cleaning: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; female 610, male 560, unknown 30; 0 missing ages; every age between 18 and 60.

**Orders** (applied in this order)

| Rule | Rows or values changed |
|---|---|
| 1. order_ts converted to datetime | all rows |
| 2. Exact duplicate rows removed | 119 rows removed |
| 3. For each order_id keep only the record with the latest order_ts (removes the earlier pending record) | 656 rows removed |
| 4. order_amount converted to numeric; unparseable values filled with round(price_inr x quantity x (1 - discount_pct / 100), 2) | 112 values filled |
| 5a. Negative delivery_minutes set to missing | 105 values |
| 5b. Missing delivery_minutes of completed orders filled with the median of the order's product category among completed orders (groupby and transform); cancelled orders left missing | 774 values filled |

Rules 4 and 5a changed 112 and 105 values, a little fewer than the raw audit counts of 120 and 106, because a few of the bad values sat on duplicate or pending rows that rules 2 and 3 had already removed. After cleaning: 13,084 - 119 - 656 = 12,309 rows (one per order_id), 11,110 completed and 1,199 cancelled, no pending rows, 0 missing order_amount, 0 missing delivery_minutes among completed orders, 1,199 missing among cancelled orders, and completed revenue of INR 2,998,668.80.

### Storage and validation

The clean tables are written to `data/clean/customers.csv`, `products.csv`, `orders.csv` and to `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). `part_a_validate.py` asserts 14 checkpoints (city and gender counts, age range, 12,309 rows, one row per order_id, 11,110 completed, 1,199 cancelled, no pending rows, missing-value counts, revenue INR 2,998,668.80) and prints exactly `ALL CHECKPOINTS PASSED`.

### SQL analysis (`sql/queries.sql`, run with `pd.read_sql_query`)

| Query | Result |
|---|---|
| Q1 WHERE then HAVING (completed orders over INR 500, customer total over INR 8,000) | 30 customers |
| Q2 three-table join, revenue by city and category | 30 rows; revenues add up to INR 2,998,668.80; largest cell Bengaluru / Electronics, INR 233,909.45 |
| Q3 scalar subquery, products priced above the average price | 11 products |
| Q4 LEFT JOIN, customers with no completed order | 25 customers |
| Q5 CTEs, customers whose total is above the average total | 550 customers; average INR 2,552.06 |
| Q6 RANK, DENSE_RANK, ROW_NUMBER per city | top customers: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows where RANK differs from ROW_NUMBER |
| Q7 daily revenue with running total | 183 daily rows; last cumulative value INR 2,998,668.80 |

Why ties make RANK differ from ROW_NUMBER: tied customers share the same RANK(), but ROW_NUMBER() still numbers them one after another, so after a tie the two columns no longer match.

### pandas cross-check and exploratory analysis

The pivot table (`pd.pivot_table(..., margins=True)`) has a grand total of INR 2,998,668.80, equal to the SQL total.

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |

Completed `order_amount`: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, IQR 232.20, lower fence -283.50 (0 outliers below), upper fence 645.30 (897 outliers above). The pair with the largest correlation in absolute value is order_amount with discount_pct (-0.095); order_amount with quantity is +0.087.

**The four questions**

- *What is this data about?* Six months (April to September 2025) of synthetic quick-commerce orders in six Indian cities: 11,110 completed orders worth INR 2,998,668.80.
- *Distributions:* order_amount is strongly right-skewed (mean 269.91 against median 135.00, skewness 4.61), because most baskets are cheap groceries and a few are electronics.
- *Outliers:* 897 orders lie above the upper fence of 645.30 and none below; they are mostly genuine high-priced electronics orders, not errors. The planted errors (duplicates, negative delivery times, unparseable amounts) were removed in cleaning.
- *Relationships:* across order_amount, discount_pct, quantity and delivery_minutes the largest correlation in absolute value is order_amount with discount_pct (-0.095), just ahead of order_amount with quantity (+0.087). Both are very weak, and a correlation of this size does not establish causation.

## Part B

### RFM table

Snapshot 2025-10-01 00:00:00, completed orders only. 1,175 customers; recency median 19 (minimum 0, maximum 180); frequency mean 9.46 (maximum 33); monetary mean INR 2,552.06 (maximum INR 14,766.55).

### K-means

Features were standardised with `StandardScaler`; `KMeans(n_clusters=K, n_init=10, random_state=42)`.

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

The inertia falls steeply from K=3 (1268.2) to K=4 (743.1) and then flattens (K=5: 617.4), so the elbow is at K=4. K=4 also has the highest silhouette score (0.526) among the allowed values 3, 4 and 5, and four segments are a number the business can act on. **Chosen K = 4.**

The inertia of K=8 is always lower than that of K=2 because every extra centroid can only bring points closer to their centre, and with one cluster per customer the inertia would reach 0. A low inertia alone therefore cannot pick K; the elbow, the silhouette score and business interpretability have to be read together.

### Segment profile

Cluster numbers are only labels and carry no ordering.

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 1 | Premium High Spenders (is_premium = True) | 138 | 24.12 | 6.77 | 7,129.70 | VIP perks, early access to electronics, priority support |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Loyalty points and bundle offers; protect delivery speed |
| 0 | Occasional Low-Value | 436 | 28.86 | 3.77 | 1,057.62 | Small coupons and basket-builder cross-sells |
| 3 | Lapsing Customers | 182 | 109.04 | 2.20 | 685.99 | Win-back discount and reminder message |

The Premium flag is true only for the segment with the highest mean monetary value. `outputs/customer_segments.csv` has the columns customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium.

PCA on the scaled RFM matrix: explained variance ratios 0.634 and 0.218, total 0.852.

### Lapse problem

Cutoff 2025-09-01. The cohort is every customer with at least one completed order before the cutoff: 1,156 customers, of whom 443 lapsed (38.32%), meaning no completed order from 2025-09-01 to 2025-09-30. Split with `train_test_split(test_size=0.2, stratify=y, random_state=42)`: 924 training rows and 232 test rows. Always predicting "not lapsed" gives 61.6% accuracy and a lapsed-class F1 of 0.

Features (all computed from completed orders dated before the cutoff, plus the clean customer table): recency, frequency, monetary, average discount_pct, cancellation rate, average delivery_minutes, most frequent product category, city, age, gender, tenure in days.

**No leakage.** No order dated on or after 2025-09-01 entered the features, and the segment labels from the clustering were not used as features. The one-hot encoder (`drop="first"`) and the standard scaler sit inside a scikit-learn pipeline and were fitted on the training rows only; no encoder, scaler or resampler saw test rows. The scaler's learned mean of frequency is 8.115, against 8.029 for the full cohort, because the scaler learned its mean from the 924 training rows only and the test rows must not influence preprocessing. No feature column contains a missing value (asserted in code).

### Tuning and test results

| Model | Grid | Best parameters | Best cross-validated F1 |
|---|---|---|---|
| Logistic regression | C in {0.01, 0.1, 1, 10, 100}, cv=5, scoring f1 | C = 100 | 0.6506 |
| Random forest (100 trees) | max_depth in {3, 5, 8} x min_samples_leaf in {5, 10, 20}, cv=5, scoring f1 | max_depth = 8, min_samples_leaf = 10 | 0.6823 |

Logistic regression chose C = 100 rather than the reference C = 1; the brief allows this, since the exact feature set differs slightly.

Untouched test set, lapsed class (confusion matrix as [[TN, FP], [FN, TP]]):

| Model | Confusion matrix | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (always not lapsed) | - | 0 | 0 | 0 |
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[112, 31], [31, 58]] | 0.652 | 0.652 | 0.652 |

**Selected model: random forest** (test F1 0.652, above the 0.55 requirement). The two models are nearly equal on the test set, but the forest is better in cross-validation (0.6823 against 0.6506) and slightly better on precision and F1, and it gives feature importances for the retention team. Missing a customer who is about to lapse costs that customer's future orders, while wrongly targeting a loyal customer costs only a small coupon, so recall matters more than precision here. Both models catch 58 of the 89 lapsing test customers.

### Class-balance experiment

Only the training set was over-sampled: every original training row was kept and randomly chosen lapsed rows were added (with replacement, `random_state=42`), taking the class counts from 570 / 354 to 570 / 570. The best logistic regression was refitted and evaluated on the original, unchanged test set.

| Logistic regression | Precision | Recall | F1 |
|---|---|---|---|
| Un-balanced | 0.644 | 0.652 | 0.648 |
| Over-sampled | 0.589 | 0.854 | 0.697 |

Recall moved the most (+0.20), while precision fell. For the retention team this means catching about 85% of the lapsing customers instead of 65%, at the price of contacting more loyal customers by mistake, which is acceptable when a win-back message is cheap.

### Interpretation, saved model and retention list

Top five random forest features: frequency 0.3072, monetary 0.2468, recency 0.1149, cancel_rate 0.0615, avg_discount 0.0580.

| Logistic regression feature | Coefficient (standardised) | Business reading |
|---|---|---|
| frequency | -1.698 | Customers who order often rarely lapse; a drop in frequency is the best early warning |
| recency | +0.040 | A longer gap since the last order raises lapse risk; the effect is small once frequency is in the model, because the two overlap |
| monetary | +0.285 | At the same order frequency, higher total spend goes with slightly higher lapse risk; this is a conditional effect, while the forest ranks spend second in importance |

The selected random forest was saved with `joblib.dump` to `outputs/lapse_model.joblib`; the reloaded model reproduces the original test predictions exactly (checked in code).

Among the 40 test-set customers in the Premium segment, the top 10 by predicted lapse probability (from 0.5025 down to 0.4497) were written to `outputs/retention_list.csv` (customer_id, lapse_probability, actually_lapsed). **0 of these 10 truly lapsed.** The Premium flag comes from the whole-window RFM, which includes the September window that defines the lapse label, so it is a post-hoc whole-window label: this retention list is illustrative and not a leakage-free evaluation. (The model's own features use only orders before the cutoff.)

## Part C

### Input guardrails and redaction

| Check | Result |
|---|---|
| Empty tickets (status skipped_empty, never sent to the model) | 4 |
| Long tickets, more than 400 estimated tokens (status flagged_long, never sent) | 4, each estimated at 734 tokens |
| Emails redacted | 55 |
| Phone numbers redacted | 91 |
| Order ids redacted | 132 |
| Total substitutions | 278 |
| Patterns found again in the redacted texts | 0 |

Patterns are applied in this order: email, phone, order id (the exact expressions and placeholders from the brief). The user message sent to the model is exactly the redacted ticket text, and an assertion in the code checks that no email, phone or order id appears in any prompt.

### Prompts

Both system prompts contain the word CLASSIFY, list the four categories (billing, delivery, technical, fraud) and require a JSON-only reply `{"category": "...", "reason": "..."}`. The zero-shot prompt (34 estimated tokens) has no examples; the few-shot prompt (117 estimated tokens) has an `EXAMPLES:` line with four labelled examples, including a fraud example.

### Resilient calls and validation

Every model call goes through a retry function with at most 3 retries per ticket and a wait of BASE_DELAY x 2**n x random.uniform(0.9, 1.15) with BASE_DELAY = 0.05 seconds; every retry is logged (ticket id, retry number, wait) in `outputs/retry_log.txt`. `mock_llm.reset()` is called before each full run. Each full run over the 156 eligible tickets had 19 transient errors, all recovered by a retry, and no ticket ended with a permanent error. Across the zero-shot and few-shot runs there were 38 retries, all with waits between 0.0452 and 0.057 seconds, inside the allowed jitter range for retry number 0 (the log file also holds the retries of the two later runs).

Validation: valid JSON with an allowed category is `classified_json`; otherwise the lowercase words are extracted with `re.findall(r"[a-z]+", text.lower())` and the reply is `classified_keyword` if exactly one allowed category word is present; anything else, including valid JSON with a category outside the list, is `flagged_invalid_output` and goes to human review (never guessed).

Status counts (identical for both prompts): 135 classified_json, 12 classified_keyword, 9 flagged_invalid_output, 4 skipped_empty, 4 flagged_long.

### Zero-shot against few-shot

| | Zero-shot | Few-shot |
|---|---|---|
| Tickets classified | 147 | 147 |
| Correct | 126 | 147 |
| Accuracy | 85.7% | 100.0% |
| Total tokens | 9,712 | 22,660 |
| Tokens per classified ticket | 66.1 | 154.1 |

Zero-shot confusion table (rows = true category, columns = predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

The few-shot prompt costs about 88 more tokens per classified ticket (154.1 against 66.1, 2.3 times the cost). In return, accuracy rises from 85.7% to 100%, because all 21 zero-shot errors were fraud tickets (11) and technical tickets (10) sent to billing. A wrong route for a fraud ticket delays the security team, so here the extra tokens are worth paying for; for a low-stakes classification or a very large volume, the cheaper zero-shot prompt may be enough.

### Budget, routing and outputs

- **Budget demo** (zero-shot, TOKEN_BUDGET = 6000, `outputs/triage_budget_demo.csv`): 59 tickets ended `budget_exhausted` (6,049 tokens used before the budget stopped the run). Those tickets go to human review.
- **Final run** (few-shot, TOKEN_BUDGET = 100000, `outputs/triage_results.csv`): 164 rows. Routing: status skipped_empty, flagged_long, flagged_invalid_output or budget_exhausted goes to `human_review` (priority NORMAL); otherwise fraud goes to `security_team` (HIGH); otherwise a premium customer (from `outputs/customer_segments.csv`) goes to `premium_desk` (HIGH); otherwise `support_queue` (NORMAL). A customer with no row in the segment file is treated as not premium.

| Route | Tickets |
|---|---|
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |

Priorities: NORMAL 119, HIGH 45.

**Rule-validated and human-reviewed.** 147 tickets (135 by JSON, 12 by keyword extraction) were classified and validated by rules and routed automatically. 17 tickets went to humans: 4 empty, 4 too long and 9 with invalid model output.

## Part D

### Chunking, retrieval and grounded answers

The five policy files are split into chunks of 40 words with an overlap of 10 words (the source file name is stored with every chunk): **15 chunks** in total (refund 4, delivery 3, escalation 3, payment 3, privacy 2). Chunks are vectorised with `TfidfVectorizer()` and ranked by cosine similarity; `retrieve(query, k=3)` returns dictionaries with source, score and text. For the query "refund policy late delivery" the top three chunks all come from `refund_policy.txt`.

TF (term frequency) measures how often a word appears in a chunk, and IDF (inverse document frequency) gives a higher weight to words that are rare across all chunks, so common words count less. A limitation is that TF-IDF matches words, not meaning: "late delivery" and "arrived after a long wait" mean the same but share no words, so a question phrased differently can miss the right chunk.

`answer(question)` retrieves three chunks and sends two messages to `mock_llm.chat`: a system message containing POLICY ASSISTANT that tells the model to answer only from the context (and to say it could not find the information otherwise), and a user message of the form `CONTEXT:` (three chunks, each prefixed with `[source file]`) followed by `QUESTION:` and the question.

### Evaluation on `policy_eval.csv` (12 answerable and 2 unanswerable questions)

| Chunking | expected_doc in top 3 (of 12) | answer contains expected_fact (of 12) | fallback "I could not find..." (of 2) |
|---|---|---|---|
| 40 words, overlap 10 | 12 | 12 | 2 |
| 40 words, overlap 0 | 12 | 10 | 2 |

With overlap 0, two questions break: "What delivery time makes a completed order eligible for a full refund?" (the sentence containing "more than 45 minutes" was cut by a chunk boundary, so the retrieved chunk held only the end of the sentence, "receives a full refund of the order amount", and the key fact sat in the neighbouring chunk) and "How soon is money returned when a payment fails but is deducted?" (the "48 hours" part fell into the next chunk). An overlap of 10 words keeps such sentences whole in at least one chunk.

### Tools

Five tools are defined in `TOOLS_SPEC` (name, one-sentence description, JSON-style parameters) and mapped by name in `TOOL_MAP`: `get_order(order_id)` (reads `data/instamart.db`), `search_policy(query)` (returns the retrieve output under `results`), `calculate_refund(order_amount, delivery_minutes)` (full amount if delivery took more than 45 minutes, otherwise 0), `issue_refund(order_id, refund_inr)` and `escalate_to_human(order_id, refund_inr)` (each appends one JSON line with a timestamp and the tool name to `outputs/refund_ledger.jsonl`). A refund above INR 1000 is escalated and not issued.

### ReAct agent: the seven scenarios

`run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)` runs the loop and returns status, final_answer, steps, llm_calls and tokens_used. Every run is appended to `outputs/agent_runs.jsonl`.

| Scenario | Status | Refund (INR) | Model calls | Tokens used | Expected |
|---|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,077 | REFUND 128.25 |
| S2 | REFUND | 135.00 | 5 | 1,077 | REFUND 135.00 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,064 | ESCALATE 1,999.00 |
| S4 | DENIED | 0 | 4 | 749 | DENIED |
| S5 | DENIED | 0 | 3 | 445 | DENIED |
| S6 | NOT_FOUND | 0 | 2 | 144 | NOT_FOUND |
| S7 | FALLBACK_MAX_ITERATIONS | 0 | 6 | 3,216 | falls back |

S7 finished all six model calls with the default `token_limit=6000`.

### Stop-safely demonstrations

| Demonstration | Status |
|---|---|
| S1 with `token_limit=200` | FALLBACK_TOKEN_LIMIT |
| S1 with `time_limit_seconds=0` | FALLBACK_TIMEOUT |
| S7 with the default limits | FALLBACK_MAX_ITERATIONS |

Every fallback returns the steps completed so far and a short human-handoff summary in `final_answer`.

### Crash and resume

After every tool observation a checkpoint is written to `state/<run_id>.json` (run id, scenario id, full message list, tokens_used, step number and the list of tools already executed). For S1, a simulated crash was raised right after the second tool execution (after its checkpoint was saved) and `resume_agent("S1_crash_demo")` continued the run. Tools executed across the crash and the resume, in order:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`

That is exactly four executions, none repeated, and the final result was `REFUND: INR 128.25 issued.`

### ReAct versus plan-and-execute

My ReAct agent asks the model for one step at a time: 5 calls for S1, 4 for S4, 3 for S5 and 2 for S6, with the prompt growing at each call (S1 used 1,077 estimated tokens). A plan-and-execute design asks for the whole step list once, so it needs fewer model calls (about 1 for the plan, plus 1 optional final answer) and fewer tokens, because the growing message history is not re-sent each time.

The difference shows up when an observation is unexpected. In S6 (NOT_FOUND) ReAct stopped after 2 calls, and in S5 (cancelled order) it answered DENIED after 3 calls, because each observation decided the next step. A plan written up front would already contain `calculate_refund` and `issue_refund`, so it would need extra checks or a replanning call to avoid acting on an order that does not exist or was never charged.

I would choose ReAct for this refund workflow. Money moves at the last steps, every step depends on what the previous tool returned, and the extra cost (about 2 to 4 more model calls per refund) is small. ReAct also fits checkpointing: the S1 crash-and-resume run continued from the saved message list without repeating a tool. Plan-and-execute would suit long, predictable workflows where model calls are expensive.

## Limitations and Next Steps

- **The retention list is illustrative, not a leakage-free evaluation.** The Premium flag comes from the whole-window RFM, which includes the September window that defines the lapse label, so it is a post-hoc whole-window label (0 of the 10 listed customers lapsed for exactly this reason). A leakage-free version would compute the segments from orders before the cutoff only.
- All data are synthetic, so the segments, lapse model and ticket results show the method, not real Instamart behaviour.
- The lapse model uses one cutoff and one test split (232 customers); a time-based rolling validation would give steadier estimates. The logistic regression best C differs from the reference run, which is within the brief's tolerance.
- The language model is a deterministic offline mock that classifies by keywords, so the 100% few-shot accuracy and the policy answers do not predict the behaviour of a real hosted model.
- Retrieval uses TF-IDF, which matches words and not meaning; embeddings would handle differently phrased questions better.
- The refund agent trusts the order database and a fixed 45-minute rule; a production version would add authentication, the 7-day request window and a human approval queue for escalated refunds.
- Next steps: rolling-window lapse validation, probability calibration, a cost-based threshold for the retention list, and evaluation of the triage and agent steps with a real hosted model kept behind the same interface.
