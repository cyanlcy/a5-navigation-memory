"""Join public features and grading keys inside the trusted evaluator only."""
import argparse
import hashlib
import json
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
def main(split):
    root=BASE/'dataset'/split
    truth=json.loads((root/'truth.json').read_text())
    rows=[]
    for row in json.loads((root/'inputs.json').read_text()):
        k=row['key'];v=json.loads((root/'features'/(k+'.json')).read_text())
        assert k==v['key'] and v['observation']['views']==row['observation']['views']
        files=[root/'rgb'/row['route']/(name+'.png') for name in ('old0','old1','current')]
        assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]==v['images']
        rows.append(dict(key=k,public={s:v[s] for s in ('observation','visual')},truth=truth[k]))
    assert len(rows)==len(truth)
    dest=root/'evaluation.json';dest.write_text(json.dumps(rows,separators=(',',':')))
    return dest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split');print(main(p.parse_args().split))
