import base64,json,os,shutil,subprocess,sys,tempfile,time,uuid,hashlib,threading
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from renopilot import config
from renopilot.llm import ClaudeCLIChat,_tool_protocol
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult,ChatGeneration
MODEL="gpt-6.1-sol"
CLI=shutil.which("codex")
OUT=ROOT/"evals/results"/("codex_bugfix_"+time.strftime("%Y%m%d_%H%M%S"))
OUT.mkdir(parents=True,exist_ok=True)
LOCK=threading.Lock()
CALLS=0

def strict(s):
    if isinstance(s,list):return [strict(v) for v in s]
    if not isinstance(s,dict):return s
    s={k:strict(v) for k,v in s.items()}
    if s.get("type")=="object" or "properties" in s:
        s["additionalProperties"]=False;s["required"]=list(s.get("properties",{}))
    return s

class CodexChat(ClaudeCLIChat):
    model:str=MODEL
    def _generate(self,messages,stop=None,run_manager=None,json_schema=None,tools=None,tool_choice=None,**kwargs):
        global CALLS
        with LOCK:
            CALLS+=1
            if CALLS>100:raise RuntimeError("Quick-check model call limit reached")
        system,blocks=self._build_turn(messages)
        if tools and tool_choice!="none":
            protocol,json_schema=_tool_protocol(tools,tool_choice);system+="\n\n"+protocol
        with tempfile.TemporaryDirectory(prefix="renopilot_check_") as td:
            tmp=Path(td)
            args=[CLI,"exec","--ignore-user-config","--ephemeral","--skip-git-repo-check","--sandbox","read-only","--model",MODEL,"-c",'model_reasoning_effort="low"',"-c","features.shell_tool=false","--json","--output-last-message",str(tmp/"reply.txt"),"--cd",td]
            texts=[]
            for n,b in enumerate(blocks):
                if b["type"]=="text":texts.append(b["text"])
                elif b["type"]=="image":
                    src=b["source"]
                    if src.get("type")!="base64":raise ValueError("Unsupported image source")
                    ip=tmp/f"image{n}.jpg";ip.write_bytes(base64.b64decode(src["data"]));args += ["--image",str(ip)]
            if json_schema:
                (tmp/"schema.json").write_text(json.dumps(strict(json_schema)),encoding="utf-8")
                args += ["--output-schema",str(tmp/"schema.json")]
            args += ["-"]
            prompt="You are a model backend for an evaluation. Do not use native tools, inspect files, browse, or execute commands. Express application tool calls only in the requested JSON response.\n\nSYSTEM:\n"+system+"\n\nCONVERSATION:\n"+"\n".join(texts)
            proc=subprocess.run(args,input=prompt,capture_output=True,text=True,encoding="utf-8",timeout=180)
            if proc.returncode or not (tmp/"reply.txt").exists():raise RuntimeError("Codex backend failed: "+(proc.stderr+proc.stdout)[-1400:])
            events=[]
            for line in proc.stdout.splitlines():
                try:events.append(json.loads(line))
                except ValueError:pass
            if any(e.get("item",{}).get("type") in ("command_execution","mcp_tool_call","web_search") for e in events):raise RuntimeError("Unexpected native tool use")
            reply=(tmp/"reply.txt").read_text(encoding="utf-8").strip()
            data=json.loads(reply) if json_schema else None
            content,calls=reply,[]
            if tools and tool_choice!="none":
                calls=[{"name":c["name"],"args":c.get("args") or {},"id":"call_"+uuid.uuid4().hex[:24],"type":"tool_call"} for c in data.get("tool_calls",[])]
                content=data.get("content") or ""
            usage=next((e.get("usage",{}) for e in reversed(events) if e.get("type")=="turn.completed"),{})
            ni,no=usage.get("input_tokens",0),usage.get("output_tokens",0)
            msg=AIMessage(content=content,tool_calls=calls,usage_metadata={"input_tokens":ni,"output_tokens":no,"total_tokens":ni+no},response_metadata={"model":MODEL,"backend":"codex_cli_quick","structured_output":data,"cost_usd":0.0})
            return ChatResult(generations=[ChatGeneration(message=msg)])

import renopilot.llm as llm
llm.get_model=lambda role:CodexChat()
from evals.harness import load_cases,run_eval
from evals.run_end_to_end import make_agent,grade
from renopilot.vocab import WORK_TYPE_OF

def checked_grade(case,result):
    ok=grade(case,result)
    state=result.output["state"]
    counts=Counter(i.item_type for i in state["style_profile"].items if WORK_TYPE_OF[i.item_type]=="loose_furniture")
    duplicate=sum(max(0,n-1) for n in counts.values())
    result.metrics["duplicate_furniture_requests"]=float(duplicate)
    if duplicate:
        result.failure_tags.append("duplicate_furniture_requests");ok=False
    print("CHECK",case.id,"furniture",dict(counts),"work",[(s.item_type,s.verdict.decision,s.verdict.rule_id) for s in state.get("suggestions",[]) if not s.product_id and s.verdict],flush=True)
    return ok

patch=subprocess.check_output(["git","diff","--","renopilot","tests"],cwd=ROOT,text=True,encoding="utf-8")
(OUT/"source_changes.patch").write_text(patch,encoding="utf-8")
meta={"backend":"codex_cli_quick","model":MODEL,"effort":"low","runs":1,"selected_cases":["e01","e02","e08"],"commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"source_patch_sha256":hashlib.sha256(patch.encode()).hexdigest(),"purpose":"limited bugfix verification, not final Claude eval","cost_recorded":False,"extra_grading":"No duplicate furniture types in these single-room scenarios"}
(OUT/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
shutil.copyfile(__file__,OUT/"runner.py")
print("RESULT_DIR",OUT,flush=True)
cases=[c for c in load_cases(ROOT/"evals/cases/end_to_end.jsonl") if c.id in meta["selected_cases"]]
rep=run_eval(make_agent("live"),cases,checked_grade,n_runs=1,name="Codex bugfix regression",revision=json.dumps(meta,sort_keys=True),resume_file=OUT/"end_to_end.jsonl")
print(rep.render(),flush=True)
for o in rep.outcomes:print("RESULT",o.case_id,o.passed,o.result.failure_tags,o.result.error,round(o.result.latency_s,1),flush=True)
print("MODEL_CALLS",CALLS,flush=True)
