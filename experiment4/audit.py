"""Recompute pilot outcomes, budget curves and pairing from source and recorded decisions."""
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
BASE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def write_table(name,rows):
    with (BASE/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    ds=BASE/'dataset/pilot';truth=read(ds/'truth.json');parts=read(ds/'split.json')
    assert len(set(sum(parts.values(),[])))==8
    assert len(truth)==48 and len(set(t['pair'] for t in truth.values()))==24
    for r in read(ds/'inputs.json'):
        v=read(ds/'features'/(r['key']+'.json'))
        assert [sha((ds/'rgb'/r['route']/(s+'.png')).read_bytes()) for s in ('old0','old1','current')]==v['images']
    budget=[];curves=[];changes=[]
    for v,suffix in [(1,''),(2,'_v2')]:
        root=BASE/('runs'+suffix);out=BASE/('pilot_results'+suffix)
        sealed=read(out/'sealed.json');lock=read(out/'lock.json');metrics=read(out/'metrics.json')
        for seed in (11,29,47):
            folder=root/str(seed)
            docs=[read(p) for p in folder.glob('*.json') if not p.name.endswith('_lock.json')]
            assert len(docs)==21
            budget.append(dict(version=v,seed=seed,calls=len(docs),code_calls=20,guide_calls=1,
                input_tokens=sum(d['input_tokens'] for d in docs),output_tokens=sum(d['output_tokens'] for d in docs),
                generation_seconds=round(sum(d['seconds'] for d in docs),2),
                fully_invalid=sum(all(x['action'] is None for x in d['rows']) for d in docs if 'rows' in d),
                max_input=max(d['input_tokens'] for d in docs),output_cap_hits=sum(d['output_tokens']==(650 if v==1 else 1200) for d in docs if 'rows' in d)))
            for start in ('direct','identify'):
                seq={}
                for arm in 'BC':
                    key=f'{seed}/{start}/{arm}'
                    src=read(folder/f'{start}_{arm}_lock.json')['curve'];seq[arm]=lock[key]
                    assert [sha(c.encode()) for c in src]==lock[key]
                    for j,h in enumerate(lock[key]):
                        rows=sealed[h]['rows'];assert len(rows)==12 and all(x['isolated'] for x in rows)
                        grouped=defaultdict(list)
                        for x in rows:
                            t=truth[x['key']];assert t['room'] in parts['holdout']
                            good=x['action']==t['correct'] and x['frame'] in t['frames']
                            assert good==x['joint'];grouped[t['pair']].append(good)
                        whole=sum(all(g) for g in grouped.values())
                        assert abs(whole/6-metrics[key]['curve'][j])<1e-9
                        curves.append(dict(version=v,seed=seed,start=start,arm=arm,round=j,whole=whole,pairs=6))
                    for j in range(4):
                        d=read(folder/f'{start}_{arm}_{j}.json')
                        assert d['parent_hash']==lock[key][j] and (sha(d['source'].encode()) if d['accepted'] else d['parent_hash'])==lock[key][j+1]
                assert seq['B'][0]==seq['C'][0]
                for a,b in zip(sealed[seq['B'][-1]]['rows'],sealed[seq['C'][-1]]['rows']):
                    assert a['key']==b['key']
                    if (a['frame'],a['action'])!=(b['frame'],b['action']):
                        t=truth[a['key']];changes.append(dict(version=v,seed=seed,start=start,key=a['key'],family=t['kind'],B_frame=a['frame'],C_frame=b['frame'],B_action=a['action'],C_action=b['action'],B_joint=a['joint'],C_joint=b['joint']))
    write_table('budget.csv',budget);write_table('transfer_curve.csv',curves);write_table('changed_actions.csv',changes)
    print(budget)
if __name__=='__main__':main()
