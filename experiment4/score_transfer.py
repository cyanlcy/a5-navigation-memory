"""After all seeds lock, score each distinct frozen pilot controller once in the isolated worker."""
import argparse
import json
import os
from collections import defaultdict
from mentor import BASE, STARTS, assess, digest, load_phase, read, save

def main(version):
    assert os.environ.get('A5_SEALED_WORKER')=='1'
    seeds=[11,29,47]
    suffix='' if version==1 else '_v2'
    root=BASE/('runs'+suffix);locks={}
    for s in seeds:
        for start in STARTS:
            for arm in 'BC':
                locks[s,start,arm]=read(root/str(s)/f'{start}_{arm}_lock.json')['curve']
    folder=BASE/('pilot_results'+suffix);folder.mkdir(exist_ok=True)
    save(folder/'lock.json',{'/'.join(map(str,k)):[digest(c) for c in v] for k,v in locks.items()})
    data=load_phase(BASE/'dataset/pilot','holdout');cache={}
    for curve in locks.values():
        for c in curve:
            h=digest(c)
            if h not in cache:
                rows,rank=assess(c,data);cache[h]=dict(rows=rows,rank=rank)
    save(folder/'sealed.json',cache)
    metrics={}
    for (s,start,arm),curve in locks.items():
        h=digest(curve[-1]);rows=cache[h]['rows']
        families=defaultdict(lambda:defaultdict(list))
        for row,d in zip(rows,data):
            t=d['truth'];families[t['kind']][t['pair']].append(row['joint'])
        metrics[f'{s}/{start}/{arm}']=dict(curve=[cache[digest(c)]['rank'][0] for c in curve],by_kind={k:[sum(map(all,v.values())),len(v)] for k,v in families.items()},actions=sum(r['ok'] for r in rows),frames=sum(r['good_frame'] for r in rows))
    save(folder/'metrics.json',metrics)
    print(json.dumps(metrics))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('version',type=int,choices=(1,2))
    main(parser.parse_args().version)
