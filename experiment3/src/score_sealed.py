"""Run inside an isolated verifier filesystem. Write one atomic result per distinct locked source hash."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from isolation import no_network
def judge():
    no_network()
    lock=json.loads(Path('/input/candidate_lock.json').read_text())
    items=json.loads(Path('/input/questions.json').read_text())
    assert len(items)==120
    for h,src in lock['sources'].items():
        assert hashlib.sha256(src.encode()).hexdigest()==h
        out=Path('/output')/(h+'.json')
        if out.exists():
            saved=json.loads(out.read_text());assert saved['hash']==h and len(saved['rows'])==120
            continue
        rows=[]
        for item in items:
            start=time.monotonic()
            with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
                try:
                    ex=subprocess.run([sys.executable,'-I','-S','/app/execute.py','--seal'],cwd=tmp,
                                      input=json.dumps(dict(source=src,row=item['public'])),text=True,capture_output=True,timeout=3)
                    if ex.returncode>0:raise RuntimeError(ex.stderr[-280:])
                    if ex.returncode<0:res=dict(action=None,frame=None,calls=[],error='resource limit',isolated=True)
                    else:res=json.loads(ex.stdout)
                except subprocess.TimeoutExpired:res=dict(action=None,frame=None,calls=[],error='timeout',isolated=True)
            assert res['isolated']
            rows.append(dict(key=item['key'],ok=res['action']==item['truth']['correct'],**res))
        partial=out.with_suffix('.tmp');partial.write_text(json.dumps(dict(hash=h,rows=rows)));partial.replace(out)
        print('scored',len(list(Path('/output').glob('*.json'))),'of',len(lock['sources']),flush=True)
if __name__=='__main__':judge()
