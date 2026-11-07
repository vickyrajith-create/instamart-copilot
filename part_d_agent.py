import json, os, sqlite3, time, uuid
from datetime import datetime
import pandas as pd
import mock_llm
from part_d_rag import retrieve

DB = os.path.join("data", "instamart.db")
LEDGER = os.path.join("outputs", "refund_ledger.jsonl")
RUNS = os.path.join("outputs", "agent_runs.jsonl")
STATE = "state"
os.makedirs("outputs", exist_ok=True)
os.makedirs(STATE, exist_ok=True)


class SimulatedCrash(Exception):
    pass


# ---------------- tools ----------------
def get_order(order_id):
    con = sqlite3.connect(DB)
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                      (order_id,)).fetchone()
    con.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}


def search_policy(query):
    return {"results": retrieve(query, 3)}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}


def _ledger(tool, order_id, refund_inr):
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"),
                            "tool": tool, "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order by id in the database.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents for relevant passages.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund amount from the order amount and delivery time.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"},
                                                    "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue a refund of at most INR 1000 and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human support lead.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]

SYSTEM = ("You are a refund AGENT. Use the tools to look up the order, check the policy, "
          "calculate the refund and then issue it or escalate it. Finish with one line starting with "
          "REFUND, ESCALATE, DENIED or NOT_FOUND.")


def save_checkpoint(run_id, scenario_id, messages, tokens, step, executed):
    with open(os.path.join(STATE, f"{run_id}.json"), "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "scenario_id": scenario_id, "messages": messages,
                   "tokens_used": tokens, "step": step, "tools_executed": executed}, f)


def _fallback(status, steps, llm_calls, tokens, why):
    return {"status": status, "final_answer": f"HANDOFF to human agent: {why} Steps so far: "
            + ", ".join(s["tool"] for s in steps), "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens}


def _loop(run_id, scenario_id, messages, steps, llm_calls, tokens, executed, max_iterations,
          token_limit, time_limit_seconds, crash_after_tool=None):
    start = time.monotonic()
    while True:
        if time.monotonic() - start >= time_limit_seconds:
            return _fallback("FALLBACK_TIMEOUT", steps, llm_calls, tokens, "time limit reached.")
        if llm_calls >= max_iterations:
            return _fallback("FALLBACK_MAX_ITERATIONS", steps, llm_calls, tokens,
                             f"no final answer after {max_iterations} model calls.")
        resp = mock_llm.chat(messages, TOOLS_SPEC)
        llm_calls += 1
        tokens += resp.usage.total_tokens
        if tokens > token_limit:
            return _fallback("FALLBACK_TOKEN_LIMIT", steps, llm_calls, tokens, "token limit exceeded.")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            answer = msg.content
            return {"status": answer.split(":")[0].strip(), "final_answer": answer, "steps": steps,
                    "llm_calls": llm_calls, "tokens_used": tokens}
        tc = msg.tool_calls[0]
        messages.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        result = TOOL_MAP[tc.name](**tc.arguments)
        steps.append({"tool": tc.name, "arguments": tc.arguments, "observation": result})
        messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                         "content": json.dumps(result, default=str)})
        executed.append(tc.name)
        save_checkpoint(run_id, scenario_id, messages, tokens, len(steps), executed)
        if crash_after_tool is not None and len(executed) == crash_after_tool:
            raise SimulatedCrash(run_id)


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tool=None):
    run_id = run_id or f"{scenario_id}_{uuid.uuid4().hex[:6]}"
    mock_llm.reset()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message}]
    result = _loop(run_id, scenario_id, messages, [], 0, 0, [], max_iterations, token_limit,
                   time_limit_seconds, crash_after_tool)
    result["run_id"] = run_id
    return result


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    with open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8") as f:
        cp = json.load(f)
    messages, executed = cp["messages"], cp["tools_executed"]
    steps = [{"tool": m["tool_calls"][0]["name"], "arguments": m["tool_calls"][0]["arguments"]}
             for m in messages if m["role"] == "assistant"]
    result = _loop(run_id, cp["scenario_id"], messages, steps, len(steps), cp["tokens_used"], executed,
                   max_iterations, token_limit, time_limit_seconds)
    result["run_id"] = run_id
    result["tools_executed"] = executed
    return result