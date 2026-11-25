"""One seed and arm per run. Every generated program is physically evaluated in fresh Python processes."""
import argparse
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
def digest(source):
    return hashlib.sha256(source.encode()).hexdigest()

def evaluate(source, data):
    results = []
    for item in data:
        start = time.monotonic()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                run = subprocess.run([sys.executable, "-I", "-S", str(HERE/"execute.py"), *(["--seal"] if os.environ.get("A5_SEALED_WORKER")=="1" else [])], cwd=tmp,
                    input=json.dumps({"source": source, "row": item["public"]}), capture_output=True,
                    text=True, timeout=3, env={"PATH": "/usr/bin:/bin"})
                if run.returncode and os.environ.get("A5_SEALED_WORKER")=="1" and not run.stdout:
                    if run.returncode>0:raise RuntimeError("candidate isolation/infrastructure failure: "+run.stderr[-350:])
                result = json.loads(run.stdout)
            except (ValueError, subprocess.TimeoutExpired):
                result = dict(frame=None, action=None, calls=[], error="worker error/timeout", seconds=time.monotonic()-start)
        results.append(dict(key=item["key"], **result, ok=result["action"] == item["truth"]["correct"]))
    pairs = {}
    for item, res in zip(data, results):
        pairs.setdefault(item["truth"]["pair"], []).append(res["ok"])
    rank = (sum(r["ok"] for r in results), sum(len(vals)==2 and all(vals) for vals in pairs.values()),
            -sum(len(r["calls"]) for r in results))
    return results, rank

def diagnostics(data, parent, child):
    output = {}
    for name, before, after in [("fixed",False,True),("new_error",True,False),("still_wrong",False,False)]:
        ids = [i for i,r in enumerate(child) if r["ok"]==after and parent[i]["ok"]==before]
        item = None
        if ids:
            i = ids[0]; row = data[i]; res = child[i]
            item = dict(target=row["public"]["observation"]["target"],views=row["public"]["observation"]["views"],
                        visual_readings={k:row["public"]["visual"][k] for k in res["calls"] if k in row["public"]["visual"]},
                        answer=res["action"], correct=row["truth"]["correct"], trace=res.get("trace"), error=res.get("error"))
        output[name] = dict(count=len(ids), example=item)
    return output

RULES = """Photograph direction at capture and target direction after walking can differ. Consider how recorded position changes the bearing. Write a complete executable Python robot controller to return to an object seen earlier.
Inputs for decide(observation, tools): observation['target'] is a category, ['views'] is two old camera poses in time order. Each has heading (degrees relative to final facing, +90 right, -90 left), pitch, right_of_end_m and forward_of_end_m (old camera relative to final position, metres), step. observation['movement'] is recorded actions. Target is hidden at endpoint; one left/right turn reveals it. Each problem is independent. Coordinate convention: if the old camera was x metres to the right of the endpoint, an object d metres ahead along a +90 old heading has final lateral coordinate x+d, or x-d for a -90 heading. Positive means turn right. The actual d is unknown. The starting method ignores x; consider a plausible estimate or visual evidence. You choose how and when to use it.
There are just TWO calls across all tools combined, including repeats. Choose only one or two readings to use. A returned None action gets zero. tools.match() gives two CLIP similarities (larger better). tools.identify() gives two old-frame visual identification logits; tools.direct() gives two RGB+motion turn logits ordered left,right. tools.locate() gives two estimated camera-to-target forward ranges in metres from RGB; 0 means not detected. tools.point() returns two entries (one per old image); each is [horizontal_fraction,vertical_fraction] at the estimated bottom of the object, or None when absent. Image top-left is (0,0). Only use tools explicitly listed below. The raw RGB values are available in observation['rgb'] as three 24x24 RGB arrays when the observation includes them. If supplied, observation['calibration'] gives the robot camera height over the floor in metres and vertical field of view in degrees. A positive pitch looks down. In the calibrated four-question layout, queried objects rest on the floor. All evidence comes from frozen readings on public RGB.
Return (frame_index:int 0 or 1, action:'left' or 'right', trace:small dict). Write plain Python functions only, no imports, type annotations, docstrings, printing, exception handling, variable attributes or dictionary methods. Standard sin,cos,tan,atan,atan2,radians,sqrt,abs,all,any,bool,dict,enumerate,float,int,len,list,max,min,range,round,sorted,sum,tuple,zip,str and arithmetic are available. Use square brackets to access fields. Optimize development accuracy, both members of a pair correct, then fewer calls. Respond with a short failure hypothesis and potential regression then ONE fenced python code block containing the entire program. Do not copy labels from examples into rules."""

def search(args, model, tok):
    data = sorted(json.loads(args.data.read_text()),key=lambda x:x["key"])
    initial = (HERE/"v0.py").read_text()
    initial_out, initial_rank = evaluate(initial,data)
    best, outcomes, rank, best_round = initial,initial_out,initial_rank,0
    history, diag = [], None
    args.out.mkdir(parents=True,exist_ok=True)
    def save(path, obj):
        tmp=path.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=1));tmp.replace(path)
    save(args.out/"r00.json",dict(source=initial,hash=digest(initial),rank=rank,rows=initial_out))
    for r in range(1,args.rounds+1):
        p = args.out/f"r{r:02d}.json"
        parent,parent_out,parent_round = (initial,initial_out,0) if args.arm=="A" else (best,outcomes,best_round)
        examples = json.loads(args.examples.read_text())
        prompt = RULES + "\nAvailable tools: "+", ".join(sorted(data[0]["public"]["visual"]))+"\nCommon example: "+json.dumps(examples)+"\nStarting correct count: "+str(initial_rank[0])+"/"+str(len(data))+"\nParent source:\n"+parent
        if args.smoke:
            prompt += ("\nInterface integration exercise on open archived routes. Implement the full calculation this time: "
                       "select the evidence frame using tools.identify() once, assume one metre horizontal range for "
                       "a first-pass estimate, and compute its signed lateral position from the camera offset and "
                       "the heading sign. Return its sign, record position in trace. When a parent already implements "
                       "this, revise the inherited implementation to address remaining errors while retaining the working calculation.")
        if args.arm!="A" and history:
            prompt += "\nPrevious scores and retention: "+json.dumps(history)
        if args.arm=="C" and diag:
            prompt += "\nLast candidate compared with its actual parent: "+json.dumps(diag)
        seed = args.seed*100+r
        if p.exists():
            result=json.loads(p.read_text())
            assert result["prompt"]==prompt and result["parent_hash"]==digest(parent) and result["seed"]==seed
        else:
            torch.manual_seed(seed)
            chat=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
            inputs=tok(chat,return_tensors="pt").to(model.device)
            n=inputs.input_ids.shape[1]
            assert n<=4800,n
            start=time.monotonic()
            with torch.inference_mode():
                output=model.generate(**inputs,max_new_tokens=1000,do_sample=True,temperature=.7,top_p=.90,pad_token_id=tok.eos_token_id)
            new=output[0,n:]; raw=tok.decode(new,skip_special_tokens=True)
            seconds=time.monotonic()-start
            blocks=re.findall(r"```(?:python|py)?\s*\n(.*?)```",raw,re.S|re.I)
            if len(blocks) == 1:
                code = blocks[0].strip()+"\n"
            elif "```" not in raw and re.search(r"^def ",raw,re.M):
                code = raw[re.search(r"^def ",raw,re.M).start():].strip()+"\n"
            else:
                code = ""
            rows,newrank=evaluate(code,data)
            result=dict(round=r,arm=args.arm,seed=seed,model=str(args.model),gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                        prompt=prompt,raw=raw,source=code,hash=digest(code),parent_hash=digest(parent),parent_round=parent_round,
                        diff="".join(difflib.unified_diff(parent.splitlines(True),code.splitlines(True))),rank=newrank,
                        input_tokens=n,output_tokens=len(new),generation_seconds=seconds,device_map={k:str(v) for k,v in model.hf_device_map.items()},rows=rows,
                        diagnostic=diagnostics(data,parent_out,rows),accepted=tuple(newrank)>rank and any(x["action"] is not None for x in rows))
            save(p,result)
        if result["accepted"]:
            best,outcomes,rank,best_round=result["source"],result["rows"],tuple(result["rank"]),r
        history.append(dict(round=r,correct=result["rank"][0],retained=result["accepted"]))
        diag=result["diagnostic"]
        print(args.arm,args.seed,r,result["rank"],best_round,flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("data","out","examples","model"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--arms",default="ABC");parser.add_argument("--seeds",nargs="+",type=int,default=[17])
    parser.add_argument("--rounds",type=int,default=2)
    parser.add_argument("--smoke",action="store_true")
    a=parser.parse_args()
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    free=torch.cuda.mem_get_info()[0]//(1024**2)
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map="auto",
        max_memory={0:f"{max(3000,free-1100)}MiB","cpu":"36GiB"},attn_implementation="sdpa",local_files_only=True).eval()
    root=a.out
    for s in a.seeds:
        for arm in a.arms:
            a.seed,a.arm,a.out=s,arm,root/f"{arm}{s}"
            search(a,model,tok)
