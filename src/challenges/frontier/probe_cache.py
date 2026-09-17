"""Which cache_control placements does OpenRouter honour for anthropic/* in an agent loop? (2026-09-17: tool_list and user_after read the whole prefix back; assistant_list only its own block; none = static prefix only.)

    uv run python src/challenges/frontier/probe_cache.py anthropic/claude-opus-4.8
"""
import json, os, sys, copy, time, urllib.request
KEY=os.environ["OPENROUTER_API_KEY"]; MODEL=sys.argv[1] if len(sys.argv)>1 else "anthropic/claude-opus-4.8"
CC={"type":"ephemeral"}
filler=" ".join(f"word{i}" for i in range(2500))   # ~5K tokens of tool output
sys_text="You are a careful assistant. " + " ".join(f"rule{i}" for i in range(1500))
tools=[{"type":"function","function":{"name":"bash","description":"run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]
def call(messages, tools_):
    body={"model":MODEL,"messages":messages,"tools":tools_,"max_tokens":50,"provider":{"only":["anthropic"],"allow_fallbacks":False}}
    req=urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",data=json.dumps(body).encode(),headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
    try:
        r=json.load(urllib.request.urlopen(req,timeout=120))
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:300]}
    u=r.get("usage",{}); d=u.get("prompt_tokens_details") or {}
    return {"prompt":u.get("prompt_tokens"),"cached":d.get("cached_tokens"),"write":d.get("cache_write_tokens"),"raw_cw":u.get("cache_creation_input_tokens")}
def base(variant):
    t=copy.deepcopy(tools); t[-1]["cache_control"]=CC
    m=[{"role":"system","content":[{"type":"text","text":sys_text,"cache_control":CC}]},
       {"role":"user","content":"List the files, then summarise."},
       {"role":"assistant","content":"Let me look.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"bash","arguments":json.dumps({"command":"ls"})}}]},
       {"role":"tool","tool_call_id":"call_1","content":filler}]
    if variant=="tool_list": m[-1]["content"]=[{"type":"text","text":filler,"cache_control":CC}]
    if variant=="assistant_list": m[-2]["content"]=[{"type":"text","text":"Let me look.","cache_control":CC}]
    if variant=="user_after":  # extra user text block after the tool result
        m.append({"role":"user","content":[{"type":"text","text":"(continue)","cache_control":CC}]})
    return m,t
for variant in ["none","tool_list","assistant_list","user_after"]:
    m,t=base(variant)
    r1=call(m,t); time.sleep(2)
    m2=m+([{"role":"assistant","content":"Done looking."},{"role":"user","content":"Now summarise in five words."}])
    r2=call(m2,t)
    print(f"{variant:15s} req1={r1}  req2={r2}", flush=True)
