"""Recompute ranks, traces, paired budget curves, regression ledgers, interventions and room-cluster uncertainty."""
import collections,csv,hashlib,json,random,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];F=BASE/'formal';OUT=BASE/'results'
def read(p):return json.loads(p.read_text())
def table(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    OUT.mkdir(exist_ok=True)
    lock=read(F/'candidate_lock.json');ds={s:read(BASE/'dataset'/s/'truth.json') for s in ('dev','hidden')}
    test={p.stem:read(p) for p in (F/'hidden_results').glob('*.json')}
    assert set(test)==set(lock['sources'])
    dev={};runs={};cost=collections.defaultdict(collections.Counter)
    for d in sorted((F/'runs').iterdir()):
        if not d.is_dir():continue
        name=d.name;docs=[read(d/f'r{i:02d}.json') for i in range(13)];runs[name]=docs
        for r,doc in enumerate(docs):
            h=doc['hash'];assert hashlib.sha256(doc['source'].encode()).hexdigest()==h
            rows={x['key']:x for x in doc['rows']};assert set(rows)==set(ds['dev'])
            assert all(x.get('isolated',False) or x.get('error') in ('worker error/timeout','timeout') for x in rows.values())
            assert all(x['ok']==(x['action']==ds['dev'][k]['correct']) for k,x in rows.items())
            count=sum(x['ok'] for x in rows.values());groups=collections.defaultdict(list)
            for k,x in rows.items():groups[ds['dev'][k]['pair']].append(x['ok'])
            assert doc['rank']==[count,sum(all(x) for x in groups.values()),-sum(len(x['calls']) for x in rows.values())]
            scores={k:x['ok'] for k,x in rows.items()}
            if h in dev:assert scores==dev[h]
            else:dev[h]=scores
            if r:
                parent=docs[doc['parent_round']];prev=dev[parent['hash']]
                assert doc['diagnostic']['fixed']['count']==sum(not prev[k] and scores[k] for k in scores)
                assert doc['diagnostic']['new_error']['count']==sum(prev[k] and not scores[k] for k in scores)
                c=cost[name[0]]
                c.update(dict(proposals=1,input_tokens=doc['input_tokens'],output_tokens=doc['output_tokens'],
                         generation_seconds=doc['generation_seconds'],dev_questions=len(rows),logical_visual_calls=-doc['rank'][2],
                         invalid_decisions=sum(x['action'] is None for x in rows.values()),all_invalid=int(all(x['action'] is None for x in rows.values()))))
    seeds=[11,29,47,71,101]
    for s in seeds:
        assert len({runs[f'{a}{s}'][1]['prompt'] for a in 'ABC'})==1
        assert len({runs[f'{a}{s}'][1]['hash'] for a in 'ABC'})==1
    hidden={}
    for h,doc in test.items():
        assert h==doc['hash'] and len(doc['rows'])==120
        rows={r['key']:r for r in doc['rows']};assert set(rows)==set(ds['hidden'])
        assert all(r['ok']==(r['action']==ds['hidden'][k]['correct']) and r['isolated'] for k,r in rows.items())
        hidden[h]={k:r['ok'] for k,r in rows.items()}
    st=lock['start']
    def measure(scores,split):
        t=ds[split];groups=collections.defaultdict(list)
        for k,v in scores.items():groups[t[k]['pair']].append(v)
        ep=read(BASE/'dataset'/split/'endpoint_pairs.json')
        return dict(correct=sum(scores.values()),both_targets=sum(all(g) for g in groups.values()),both_endpoints=sum(all(scores[k] for k in g) for g in ep.values()),
                    cross=sum(v for k,v in scores.items() if t[k]['relation']=='cross'),same=sum(v for k,v in scores.items() if t[k]['relation']=='same'))
    per=[]
    for name in sorted(runs):
        for n in (1,3,6,12):
            h=lock['checkpoints'][name+'/'+str(n)];z=lock['runs'][name]['best_by_round'][n]
            per.append(dict(arm=name[0],seed=int(name[1:]),budget=n,selected=z,
                            **{'dev_'+k:v for k,v in measure(dev[h],'dev').items()},**{'hidden_'+k:v for k,v in measure(hidden[h],'hidden').items()}))
    table('paired_seeds.csv',per)
    curves=[]
    for n in (1,3,6,12):
        for arm in 'ABC':
            pool=[r for r in per if r['budget']==n and r['arm']==arm]
            curves.append(dict(arm=arm,budget=n,**{k:statistics.mean(r[k] for r in pool) for k in per[0] if k.startswith(('dev_','hidden_'))}))
    table('curves.csv',curves)
    end=[r for r in curves if r['budget']==12]
    for r in end:
        r['dev_gain_pp']=(r['dev_correct']-sum(dev[st].values()))*100/120
        r['hidden_gain_pp']=(r['hidden_correct']-sum(hidden[st].values()))*100/120
        r['overfit_gap_pp']=r['dev_gain_pp']-r['hidden_gain_pp']
    table('main.csv',end)
    rooms=collections.defaultdict(list)
    for k,t in ds['hidden'].items():rooms[t['room']].append(k)
    diffs=[];rng=random.Random(20261124)
    for a,b in [('C','B'),('B','A'),('C','A')]:
        totals={}
        for room,keys in rooms.items():
            totals[room]=sum(int(hidden[lock['checkpoints'][f'{a}{s}/12']][k])-int(hidden[lock['checkpoints'][f'{b}{s}/12']][k]) for s in seeds for k in keys)
        z=[];keys=list(rooms)
        for _ in range(10000):
            picked=rng.choices(keys,k=len(keys))
            z.append(100*sum(totals[k] for k in picked)/(5*sum(len(rooms[k]) for k in picked)))
        z.sort();diffs.append(dict(comparison=a+'-'+b,difference_pp=100*sum(totals.values())/(120*5),low=z[249],high=z[9749],rooms=len(keys)))
    table('paired_room_intervals.csv',diffs)
    ledger=[];ret=[]
    for name,j in lock['runs'].items():
        idx=j['chain'];hs=j['hashes']
        for a,b,x,y in zip(idx,idx[1:],hs,hs[1:]):
            ledger.append(dict(run=name,parent=a,child=b,dev_fixed=sum(not dev[x][k] and v for k,v in dev[y].items()),
                dev_broken=sum(dev[x][k] and not v for k,v in dev[y].items()),hidden_fixed=sum(not hidden[x][k] and v for k,v in hidden[y].items()),
                hidden_broken=sum(hidden[x][k] and not v for k,v in hidden[y].items())))
        for split,scores in [('dev',dev),('hidden',hidden)]:
            keep=collections.Counter()
            for k,initial in scores[st].items():
                if initial:continue
                vals=[scores[h][k] for h in hs]
                if not any(vals):continue
                after=vals[vals.index(True):]
                keep['ever_fixed']+=1
                keep['always_retained' if all(after) else ('lost_then_recovered' if after[-1] else 'lost_by_final')]+=1
            ret.append(dict(run=name,split=split,accepted_chain='>'.join(map(str,idx)),**{k:keep[k] for k in ('ever_fixed','always_retained','lost_then_recovered','lost_by_final')}))
    table('repair_ledger.csv',ledger);table('retention.csv',ret)
    ab=[];factor=[]
    for name,rec in lock['variants'].items():
        final=hidden[rec['final']];docs=runs[name]
        for q in rec['single']:
            cut=hidden[q['without']];old=hidden[docs[q['parent']]['hash']];new=hidden[docs[q['child']]['hash']]
            ab.append(dict(run=name,parent=q['parent'],child=q['child'],full=sum(final.values()),without=sum(cut.values()),
                helped=sum(final[k] and not cut[k] for k in final),hurt=sum(not final[k] and cut[k] for k in final),
                originally_fixed_and_still_needed=sum(not old[k] and new[k] and final[k] and not cut[k] for k in final)))
        z=rec['factorial']
        if z:
            factor.append(dict(run=name,early=z['early'],late=z['late'],full=sum(final.values()),without_early=sum(hidden[z['without_early']].values()),
                without_late=sum(hidden[z['without_late']].values()),without_both=sum(hidden[z['without_both']].values())))
    if ab:table('ablation.csv',ab)
    if factor:table('factorial.csv',factor)
    fixed=[dict(method=n,**measure(hidden[h],'hidden')) for n,h in lock['baselines'].items()]
    table('fixed.csv',fixed);table('cost.csv',[dict(arm=a,**cost[a]) for a in 'ABC'])
    summary=dict(rooms=len(rooms),seeds=seeds,baseline_dev=sum(dev[st].values()),baseline_hidden=sum(hidden[st].values()),
                 arms=end,intervals=diffs,stacked=factor,study_cost=sum(c['proposals'] for c in cost.values()),distinct_hidden=len(hidden))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=1))
    print(json.dumps(summary))
if __name__=='__main__':main()
