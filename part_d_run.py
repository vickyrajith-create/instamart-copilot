import json, os
import pandas as pd
import part_d_agent as ag
from part_d_agent import run_agent, resume_agent, SimulatedCrash

for p in (ag.LEDGER, ag.RUNS):
    if os.path.exists(p):
        os.remove(p)

sc = pd.read_csv("data/raw/agent_scenarios.csv")
print("=== 7 scenarios ===")
print("id | status | refund | calls | tokens | expected")
for _, r in sc.iterrows():
    res = run_agent(r["message"], r["scenario_id"])
    refund = next((s["observation"]["refund_inr"] for s in res["steps"]
                   if s["tool"] in ("issue_refund", "escalate_to_human")), 0)
    print(f'{r["scenario_id"]} | {res["status"]} | {refund} | {res["llm_calls"]} | {res["tokens_used"]} | '
          f'{r["expected_outcome"]} {r["expected_refund_inr"]}')
    with open(ag.RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps({"scenario_id": r["scenario_id"], "status": res["status"],
                            "final_answer": res["final_answer"], "llm_calls": res["llm_calls"],
                            "tokens_used": res["tokens_used"], "steps": res["steps"]}, default=str) + "\n")

s1 = sc[sc["scenario_id"] == "S1"].iloc[0]
d1 = run_agent(s1["message"], "S1", token_limit=200)
print("\nS1 with token_limit=200 ->", d1["status"])
d2 = run_agent(s1["message"], "S1", time_limit_seconds=0)
print("S1 with time_limit_seconds=0 ->", d2["status"])

print("\n=== crash and resume (S1) ===")
try:
    run_agent(s1["message"], "S1", run_id="S1_crash_demo", crash_after_tool=2)
except SimulatedCrash:
    print("simulated crash after the 2nd tool execution")
final = resume_agent("S1_crash_demo")
print("tools executed across crash + resume:", final["tools_executed"])
print("final status:", final["status"], "|", final["final_answer"])