"""Paired guide evolution and transfer. The runner loads only visible phase data during search."""
import argparse
import difflib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'experiment3/src'))
from search import evaluate,digest,RULES

GUIDE="Reproduce a failure. Compare its execution with a successful case. Identify the first divergence. Change one cause and check the regression on an earlier success."
INITIAL=(BASE.parent/'experiment3/src/v0.py').read_text()
STARTS={'identify':"""def decide(observation, tools):
    s = tools.identify()
    f = int(s[1] > s[0])
    return f, 'right' if observation['views'][f]['heading'] > 0 else 'left', {}
""",'direct':"""def decide(observation, tools):
    scores = tools.direct()
    return 0, 'right' if scores[1] > scores[0] else 'left', {}
"""}
def read(p):return json.loads(p.read_text())
def save(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=1));tmp.replace(p)

def assess(source,data):
    rows,_=evaluate(source,data)
    groups=defaultdict(lambda:defaultdict(list))
    for r,d in zip(rows,data):
        t=d['truth']
        r['good_frame']=r['frame'] in t['frames']
        r['joint']=r['ok'] and r['good_frame']
        groups[t['kind']][t['pair']].append(r['joint'])
    assert len(groups)==3 and all(len(x)==2 for ps in groups.values() for x in ps.values())
    rank=[sum(sum(all(x) for x in ps.values())/len(ps) for ps in groups.values())/3,
          sum(r['ok'] for r in rows),-sum(len(r['calls']) for r in rows)]
    return rows,rank

def feedback(rows,old,data):
    summary=defaultdict(list)
    for r,b,d in zip(rows,old,data):
        key='fixed' if r['joint'] and not b['joint'] else 'regressed' if b['joint'] and not r['joint'] else 'still_wrong' if not r['joint'] else 'stable'
        t=d['truth']
        summary[key].append(dict(family=t['kind'],valid_frame=r['good_frame'],action_ok=r['ok'],calls=r['calls'],error=r.get('error'),trace=r.get('trace'),actual_frame=r['frame'],valid_frames=t['frames'],actual_turn=r['action'],expected_turn=t['correct'],views=d['public']['observation']['views'],readings={name:d['public']['visual'][name] for name in r['calls']}))
    return {k:dict(count=len(v),example=v[0]) for k,v in summary.items()}

def load_phase(root,phase):
    rooms=read(root/'split.json')[phase];truth=read(root/'truth.json')
    rows=[]
    for k,t in sorted(truth.items()):
        if t['room'] in rooms:
            v=read(root/'features'/(k+'.json'))
            rows.append(dict(key=k,truth=t,public={s:v[s] for s in ('observation','visual')}))
    return rows

def generate(prompt,seed,limit=1200):
    chat=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
    inp=tok(chat,return_tensors='pt').to(model.device); n=inp.input_ids.shape[1]
    assert n<=4800,n
    torch.manual_seed(seed);start=time.monotonic()
    with torch.inference_mode():
        out=model.generate(**inp,max_new_tokens=limit,temperature=.7,top_p=.9,do_sample=True,pad_token_id=tok.eos_token_id)[0,n:]
    return dict(prompt=prompt,raw=tok.decode(out,skip_special_tokens=True),input_tokens=n,output_tokens=len(out),seconds=time.monotonic()-start,seed=seed)

def code_call(folder,name,seed,guide,source,data,diag):
    rows,rank=assess(source,data)
    prompt=RULES.split('Optimize development accuracy')[0]+'All tool methods take ZERO arguments. Use at most TWO tool calls in the whole function. Save each result to a variable. Output concise code, under 30 lines, without prose or comments. Keep the working parts of the parent. Score first on whole paired tasks (valid evidence and action on BOTH), then on action accuracy and fewer calls. Return one fenced Python program.\nDebugging guide: '+guide+'\nParent source:\n'+source+'\nDevelopment feedback: '+json.dumps(diag or feedback(rows,rows,data))
    p=folder/(name+'.json')
    if p.exists():
        v=read(p);assert v['prompt']==prompt and v['seed']==seed
        return v
    v=generate(prompt,seed)
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',v['raw'],re.S|re.I)
    code=blocks[0].strip()+'\n' if len(blocks)==1 else (v['raw'].strip()+'\n' if v['raw'].lstrip().startswith('def ') and '```' not in v['raw'] else '')
    new,score=assess(code,data)
    v.update(source=code,parent=source,parent_hash=digest(source),diff=''.join(difflib.unified_diff(source.splitlines(True),code.splitlines(True))),rank=score,parent_rank=rank,rows=new,feedback=feedback(new,rows,data),accepted=score>rank and any(r['action'] for r in new))
    save(p,v);return v

def run(seed):
    folder=BASE/'runs_v2'/str(seed);folder.mkdir(parents=True,exist_ok=True)
    learn,cal,adapt=[load_phase(BASE/'dataset/pilot',p) for p in ('learn','calibrate','adapt')]
    parent=INITIAL;diag=None;records=[]
    for j in range(2):
        v=code_call(folder,'learn'+str(j),seed*100+j,GUIDE,parent,learn,diag)
        if v['accepted']:parent=v['source']
        diag=v['feedback'];records.append(dict(feedback=diag,diff=v['diff'][:1800],accepted=v['accepted']))
    p=folder/'guide.json'
    prompt='Read the two code revisions and their results. Append exactly ONE short imperative sentence to the existing guide that would prevent a concrete error in a future code repair. Output only the added sentence, maximum 35 words. Be specific about what to inspect or check when testing a patch. General procedures only. No code, formulas, identifiers, coordinates, object names or example answers. Guide: '+GUIDE+'\nFeedback: '+json.dumps(records)
    if p.exists():
        meta=read(p);assert meta['prompt']==prompt
    else:
        meta=generate(prompt,seed*100+11,160)
        text=meta['raw'].strip()
        words={r['public']['observation']['target'] for r in learn+cal+adapt}
        valid=bool(text) and len(text.split())<=35 and len(re.findall(r'[.!?](?:\s|$)',text))<=1 and not re.search(r'[`{}=\d]|FloorPlan|\b(?:left|right)\b',text,re.I) and not any(re.search(r'\b'+re.escape(w)+r'\b',text,re.I) for w in words)
        meta.update(guide=GUIDE+' '+text,added=text,valid=valid);save(p,meta)
    challenger=meta['guide'] if meta['valid'] else GUIDE
    a,b=[code_call(folder,'cal_'+tag,seed*100+17,g,INITIAL,cal,None) for tag,g in [('parent',GUIDE),('candidate',challenger)]]
    kept=meta['valid'] and b['rank']>a['rank'] and b['accepted']
    chosen=challenger if kept else GUIDE
    save(folder/'guide_lock.json',dict(retained=kept,guide=chosen,sha=digest(chosen),parent=a['rank'],candidate=b['rank']))
    for start,initial in STARTS.items():
        for arm,g in [('B',GUIDE),('C',chosen)]:
            parent=initial;diag=None;seq=[parent]
            for j in range(4):
                v=code_call(folder,f'{start}_{arm}_{j}',seed*100+30+j,g,parent,adapt,diag)
                if v['accepted']:parent=v['source']
                diag=v['feedback'];seq.append(parent)
            save(folder/f'{start}_{arm}_lock.json',dict(curve=seq,guide_hash=digest(g)))
    print('locked seed',seed,'guide retained',kept,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('seeds',nargs='+',type=int);args=ap.parse_args()
    assert os.environ.get('A5_SEALED_WORKER')=='1'
    path=os.environ['A5_CODER_MODEL']
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval()
    for seed in args.seeds:run(seed)
