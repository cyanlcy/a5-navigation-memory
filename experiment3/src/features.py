"""Cache frozen public visual tools, without loading any instance masks, correct actions or scene manifests."""
import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, Qwen2_5_VLForConditionalGeneration
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def text(s):return dict(type='text',text=s)
def image():return dict(type='image')
def main(a):
    src=BASE/'dataset'/a.split
    data=json.loads((src/'inputs.json').read_text())
    weights=ROOT/'models/qwen-vl';clipdir=ROOT/'models/clip'
    proc=AutoProcessor.from_pretrained(weights,local_files_only=True)
    # Leave room for graphics and jobs already running on the shared cards.
    free=torch.cuda.mem_get_info()[0]//(1024**2)
    cap=max(3200,min(20000,free-1800))
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(weights,dtype=torch.bfloat16,device_map='auto',
        max_memory={0:f'{cap}MiB','cpu':'40GiB'},attn_implementation='sdpa',local_files_only=True).eval()
    cp=AutoProcessor.from_pretrained(clipdir,local_files_only=True);clip=CLIPModel.from_pretrained(clipdir,local_files_only=True).eval()
    torch.set_num_threads(4)
    out=src/'features';out.mkdir(exist_ok=True)
    for row in data[a.shard::a.shards]:
        p=out/(row['key']+'.json')
        if p.exists():continue
        start=time.monotonic()
        files=[src/'rgb'/row['route']/(n+'.png') for n in ('old0','old1','current')]
        ims=[Image.open(f).convert('RGB') for f in files]
        obs=row['observation'];word=obs['target']
        def batch(parts,photos):
            prompt=proc.apply_chat_template([dict(role='user',content=parts)],tokenize=False,add_generation_prompt=True)
            return proc(text=[prompt],images=photos,return_tensors='pt').to('cuda:0')
        def binary(parts,photos,options):
            b=batch(parts,photos);ids=[proc.tokenizer.encode(t,add_special_tokens=False) for t in options]
            assert all(len(i)==1 for i in ids)
            return model(**b,use_cache=False,logits_to_keep=1).logits[0,-1,[i[0] for i in ids]].float().tolist()
        pair=[text('Photographs from a home. First old photo:'),image(),text('Second old photo:'),image()]
        with torch.inference_mode():
            v=clip.get_image_features(**cp(images=ims[:2],return_tensors='pt'))
            t=clip.get_text_features(**cp(text='a photo of a '+word,return_tensors='pt'))
            match=(torch.nn.functional.normalize(v,dim=-1)@torch.nn.functional.normalize(t,dim=-1)[0]).tolist()
            identify=binary(pair+[text('Which photograph actually shows the '+word+'? Answer first or second.')],ims[:2],['first','second'])
            direct=binary(pair+[text('Endpoint facing forward:'),image(),text('Old camera poses relative to the endpoint: '+
                json.dumps(obs['views'])+'; movement: '+str(obs['movement'])+'. Final camera pitch down in degrees: '+str(obs.get('endpoint_pitch',0))+'. Which 90 degree turn reveals the '+word+
                '? Answer left or right.')],ims,['left','right'])
            parts=pair+[text('Find the '+word+' in the old photographs. In which single photo is it most clearly visible? '
                'Give a point at the bottom center of the visible object, just inside its lower boundary, x and y as percentages 0 to 100 from top-left. Estimate horizontal distance from that camera to the object in metres. '
                'Answer only one JSON object with numeric fields frame (1 or 2), x, y, meters.')]
            b=batch(parts,ims[:2]);n=b.input_ids.shape[1]
            gen=model.generate(**b,max_new_tokens=94,do_sample=False)
            raw=proc.tokenizer.decode(gen[0,n:],skip_special_tokens=True)
        ranges=[0.,0.];point=None
        try:
            d=json.loads(re.search(r'\{[^{}]*\}',raw)[0]);i=int(d['frame'])-1;x=float(d['x']);y=float(d['y']);m=float(d['meters'])
            if i in (0,1) and 0<=x<=100 and 0<=y<=100 and 0<m<=15:
                ranges[i]=m;point=[i,x,y]
        except (AttributeError,TypeError,ValueError,KeyError,IndexError):pass
        # Compact raw RGB grid for direct pixel operations in generated code; the four tools see original resolution.
        rgb=[np.asarray(im.resize((24,24),Image.Resampling.BILINEAR)).tolist() for im in ims]
        result=dict(key=row['key'],observation=dict(**obs,rgb=rgb),visual=dict(match=match,identify=identify,direct=direct,locate=ranges),
                    point=point,raw=raw,images=[sha(f) for f in files],seconds=time.monotonic()-start)
        coords=[None,None]
        if point is not None:coords[point[0]]=[point[1]/100,point[2]/100]
        result['visual']['point']=coords
        temp=p.with_suffix('.tmp');temp.write_text(json.dumps(result,separators=(',',':')));temp.replace(p)
        print(row['key'],round(result['seconds'],2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    main(p.parse_args())
