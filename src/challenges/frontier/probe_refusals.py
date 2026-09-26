"""First-turn refusal rate of an Anthropic model on the frontier items (system + instruction + bash tool).

    uv run python src/challenges/frontier/probe_refusals.py anthropic/claude-fable-5 configs/bench/terminal_bench_2_1_frontier.yml configs/bench/swe-bench-mini-frontier.yml

2026-09-17 claude-fable-5: 10/82 terminal tasks refused ("violative cyber content"), 0/40 swe.
"""
import json, os, sys, urllib.request, concurrent.futures as cf
from pathlib import Path
sys.path.insert(0, "src")
from configs import TaskConfig
from configs.paths import resolve_task_fn
KEY=os.environ["OPENROUTER_API_KEY"]; MODEL=sys.argv[1]
SYSTEM="You are a helpful assistant attempting to submit the best possible answer. You have several tools available to help with finding the answer. You will see the result of tool calls right after sending the message. If you need to perform multiple actions, you can always send more messages with additional tool calls. Do some reasoning before your actions, describing what tool calls you are going to use and how they fit into your plan.\n\nWhen you have completed the task and have an answer, call the submit() tool to report it."
TOOLS=[{"type":"function","function":{"name":"bash","description":"Use this function to execute bash commands.","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}},
       {"type":"function","function":{"name":"submit","description":"Submit an answer for evaluation.","parameters":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}}}]
def probe(sample):
    body={"model":MODEL,"max_tokens":300,"tools":TOOLS,"provider":{"only":["anthropic"],"allow_fallbacks":False},
          "messages":[{"role":"system","content":SYSTEM},{"role":"user","content":str(sample.input)}]}
    req=urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",data=json.dumps(body).encode(),headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
    try:
        r=json.load(urllib.request.urlopen(req,timeout=180)); ch=r["choices"][0]
        return sample.id, ch.get("finish_reason"), ch.get("native_finish_reason"), (ch["message"].get("refusal") or "")[:80]
    except Exception as e:
        return sample.id, "ERROR", str(e)[:120], ""
for yml in sys.argv[2:]:
    b=TaskConfig.from_cli(Path(yml)); t=resolve_task_fn(b.task)(**b.task_args)
    with cf.ThreadPoolExecutor(8) as ex: rows=list(ex.map(probe, list(t.dataset)))
    refused=[r for r in rows if r[1]=="content_filter" or r[2]=="refusal"]
    print(f"== {b.name} {MODEL}: {len(refused)}/{len(rows)} refused; other finish reasons: {sorted({r[1] for r in rows})}")
    for r in refused: print("  ", r[0], r[3])
    for r in rows:
        if r[1]=="ERROR": print("  ERR", r[0], r[2])
