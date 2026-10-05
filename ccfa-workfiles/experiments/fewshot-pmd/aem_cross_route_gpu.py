"""Exact AEM fixed-boundary/class rescoring. GPU branch; original CPU experiment preserved."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import sys
import time


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,r):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf-8')


def context():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','smoke','native','cross'])
    ap.add_argument('--root',default='/hy-tmp');ap.add_argument('--threads',type=int,default=1)
    args=ap.parse_args();root=Path(args.root);here=root/'ccfa-workfiles/experiments/fewshot-pmd'
    out=here/'aem_cross_route_runtime'/'gpu';out.mkdir(parents=True,exist_ok=True)
    jobs=[json.loads(s) for s in (here/'g1_runtime/jobs.jsonl').read_text().splitlines()]
    jobs=[j for j in jobs if j['method']=='aem' and j['recipe']=='coffee' and j['K'] in ['2','full'] and j['seed'] in [0,1]]
    assert len(jobs)==4 and all(j['test']==jobs[0]['test'] for j in jobs)
    return args,root,here,out,jobs


def initialize(root,out,threads):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['MAX_JOBS']='2'
    import torch
    torch.set_num_threads(threads);torch.set_num_interop_threads(1);torch.manual_seed(0)
    from torch.utils.cpp_extension import load
    build=out/'nms_build';build.mkdir(exist_ok=True)
    nms=load(name='nms_1d_cpu',sources=[str(root/'code/AEM/libs/utils/csrc/nms_cpu.cpp')],
             extra_cflags=['-fopenmp'],extra_ldflags=['-fopenmp'],build_directory=str(build),verbose=False)
    sys.modules['libs.utils.nms_1d_cpu']=nms
    sys.path.insert(0,str(root/'code/AEM'));os.chdir(root/'code/AEM')
    import open_clip
    from safetensors import safe_open
    conf=open_clip.get_model_config('EVA02-L-14-336')
    import open_clip.model as clip_impl
    custom=conf.pop('custom_text',False)
    clip=(clip_impl.CustomTextCLIP if custom else clip_impl.CLIP)(**conf)
    weights=next(Path('/root/.cache/huggingface/hub/models--timm--eva02_large_patch14_clip_336.merged2b_s6b_b61k').glob('snapshots/*/open_clip_model.safetensors'))
    with safe_open(str(weights),framework='pt',device='cpu') as f:
        state={k:f.get_tensor(k) for k in f.keys()}
    clip.load_state_dict(state,strict=True);del state
    clip=clip.cuda().eval();tok=open_clip.get_tokenizer('EVA02-L-14-336');pre=None
    from libs.datasets.clip_encode import action_tokenize_egoper
    from libs.utils import fix_random_seed
    fix_random_seed(0,include_cuda=True)
    ann=read(root/'code/AEM/data/egoper/annotation.json')
    tokens=action_tokenize_egoper((clip,tok,pre),'coffee',ann['coffee'],device='cuda:0')
    return torch,(clip,tok,pre),tokens


def model_data(j,clip):
    import torch
    from libs.core import load_config
    from libs.modeling import make_meta_arch
    from libs.datasets import make_dataset
    cfg=copy.deepcopy(load_config(j['config']));cfg['devices']=['cuda:0'];cfg['loader']['num_workers']=0
    cfg['model']['backbone_type']='convTransformer'
    m=make_meta_arch(cfg['model_name'],**cfg['model']).cuda().eval()
    cp=torch.load(j['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    assert int(cp['epoch'])==135
    m.load_state_dict({k.removeprefix('module.'):v for k,v in cp['state_dict'].items()},strict=True)
    del cp
    ds=make_dataset(cfg['dataset_name'],False,clip,cfg['test_split'],False,**cfg['dataset'])
    assert ds.data_list==j['test']
    return m,ds


def route_arrays(result):
    import numpy as np
    n=len(result['frame_error_label']);pred=result['frame_action_pred'].cpu().numpy()[:n]
    seg=result['segments'].cpu().numpy().astype(np.int64);cls=result['segment_action_pred'].cpu().numpy().astype(np.int64)
    coverage=np.zeros(n,int);rp=np.zeros(n,int)
    for (a,b),c in zip(seg,cls):
        assert 0<=a<=b<n and c>=0
        coverage[a:b+1]+=1;rp[a:b+1]=c
    assert (coverage==1).all() and np.array_equal(rp,pred)
    return dict(segments=seg,segment_action_pred=cls,pred_action=pred,
                gt_action=result['frame_action_label'].cpu().numpy(),labels=(result['frame_error_label'].cpu().numpy()>0).astype(np.int64))


def rescore(m,video,route,clip,tokens):
    import torch
    # Same computations and order as the upstream inference after segment_video.
    fpn,mask,_,_,_=m.extract_feature([video])
    seg=torch.from_numpy(route['segments']);classes=torch.from_numpy(route['segment_action_pred'])
    assert classes.min()>=0 and classes.max()<len(tokens)
    feat=m.generate_segment_features(fpn,mask,[seg])
    assert all(len(f)>0 for f in feat)
    enhanced=m.effect_model.inference(feature_segments=feat,action_labels=[classes],device=m.device)
    visual=torch.stack([s.mean(dim=0) for s in enhanced],dim=0)
    sim=m.action_model.inference(clip_model=clip,segment_vis_feature=visual,action_tokens=tokens,action_labels=[classes],device=m.device)
    return sim.cpu().numpy().reshape(-1)


def frame_scores(route,sim):
    import numpy as np
    raw=np.zeros(len(route['labels']),np.float32)
    for (a,b),s in zip(route['segments'],sim):raw[a:b+1]=1-float(s)
    return dict(raw_all=raw,scores=raw*(route['pred_action']!=0),segment_action_sim=sim)


def compare(j,vid,arrays,here):
    import numpy as np
    old=np.load(here/'g1_runtime/scores/aem'/j['id']/(vid+'.npz'))
    assert np.array_equal(arrays['labels'],old['labels'])
    delta=float(np.max(np.abs(arrays['scores']-old['scores'])))
    zero=int(np.sum((arrays['scores']==0)!=(old['scores']==0)))
    report=dict(video=vid,max_score_difference=delta,zero_mask_changed_frames=zero)
    archive=here/'g1_runtime/aem_audit'/j['id']/(vid+'.npz')
    if archive.exists():
        a=np.load(archive);report['route_changed_frames']=int(np.sum(a['pred_action']!=arrays['pred_action']))
        assert report['route_changed_frames']==0,report
    assert delta<=1e-4 and zero==0,report
    return report


def registration(root,here,out,jobs):
    import yaml
    paths=[here/'aem_cross_route_gpu.py',here/'aem-cross-route-protocol.md',here/'aem-cross-route-gpu-amendment.md',here/'g1_runtime/jobs.jsonl']
    paths+=list((root/'code/AEM/libs').rglob('*.py'))+[root/'code/AEM/eval.py',root/'code/AEM/libs/utils/csrc/nms_cpu.cpp']
    data=root/'code/AEM/data/egoper';paths+=[data/'annotation.json',data/'coffee/test.txt',data/'coffee/training.txt']
    for j in jobs:
        cfg=yaml.safe_load(Path(j['config']).read_text())
        assert cfg['val_split'] is None and cfg['model']['input_dim']==256
        support=Path(cfg['dataset']['support_file']);assert support.read_text().split()==j['support_ids']
        paths+=[Path(j['checkpoint']),Path(j['config']),support]
        paths+=list((here/'g1_runtime/scores/aem'/j['id']).glob('*.npz'))
        paths+=list((here/'g1_runtime/aem_audit'/j['id']).glob('*.npz'))
    paths+=[data/'coffee/vc_v_features_10fps'/(v+'.npy') for v in jobs[0]['test']]
    clipcache=Path('/root/.cache/huggingface/hub/models--timm--eva02_large_patch14_clip_336.merged2b_s6b_b61k')
    weights=list(clipcache.glob('snapshots/*/open_clip_model.safetensors'));assert len(weights)==1
    paths+=weights
    hashes={str(p):sha(p) for p in sorted(set(paths))}
    reg=out/'registered.json'
    if reg.exists():
        old=read(reg)['hashes']
        if old!=hashes:
            amendment=read(out/'engineering_amendment.json')
            changed={p for p in hashes if hashes[p]!=old.get(p)}
            assert changed==set(amendment['changes'])
            assert all(amendment['changes'][p]=={'old':old[p],'new':hashes[p]} for p in changed)
    else:save(reg,dict(created_unix=time.time(),hashes=hashes,jobs=jobs,device='cuda:0',threads=1))
    return hashes


def main():
    args,root,here,out,jobs=context()
    if args.stage=='prepare':
        h=registration(root,here,out,jobs);print('registered',len(h),'inputs',flush=True);return
    assert (out/'registered.json').exists()
    registration(root,here,out,jobs)
    torch,clip,tokens=initialize(root,out,args.threads)
    import numpy as np
    start=time.time();reports={}
    smoke=args.stage=='smoke'
    for j in jobs:
        if not smoke:assert (out/'smoke.json').exists()
        m,ds=model_data(j,clip);rr=[]
        for i in range(1 if smoke else len(ds)):
            video=ds[i];vid=video['video_id'];tic=time.time()
            if args.stage in ['smoke','native']:
                with torch.no_grad():
                    result=m.inference([video],clip,tokens)[vid]
                    route=route_arrays(result);sim=result['segment_action_sim'].cpu().numpy().reshape(-1)
                    scores=frame_scores(route,sim);row=compare(j,vid,{**route,**scores},here)
                    if smoke:
                        again=rescore(m,video,route,clip,tokens)
                        err=float(np.max(np.abs(again-sim)));assert err<=1e-6,err
                        row['same_route_max_difference']=err
                dest=out/('smoke_native' if smoke else 'native')/j['id']/(vid+'.npz')
                dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest,**route,**scores)
            else:
                other='full' if j['K']=='2' else '2'
                routejob=f"aem_coffee_k{other}_s{j['seed']}"
                route=dict(np.load(out/'native'/routejob/(vid+'.npz')))
                own=dict(np.load(out/'native'/j['id']/(vid+'.npz')))
                assert np.array_equal(route['labels'],own['labels']) and np.array_equal(route['gt_action'],own['gt_action'])
                with torch.no_grad():sim=rescore(m,video,route,clip,tokens)
                scores=frame_scores(route,sim)
                dest=out/'cross'/f"s{j['seed']}_theta{j['K']}_route{other}"/(vid+'.npz')
                dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest,**{k:route[k] for k in ('segments','segment_action_pred','pred_action','gt_action','labels')},**scores)
                row=dict(video=vid)
            row['seconds']=time.time()-tic;rr.append(row)
            print(args.stage,j['id'],i+1,len(ds),vid,round(row['seconds'],2),flush=True)
        reports[j['id']]=rr;del m,ds
        save(out/(args.stage+'_progress.json'),dict(rows=reports,complete=False))
    if smoke:
        # Exercise actual foreign route on all models without evaluating AUC.
        for j in jobs:
            m,ds=model_data(j,clip);v=ds[0];other='full' if j['K']=='2' else '2'
            route=dict(np.load(out/'smoke_native'/f"aem_coffee_k{other}_s{j['seed']}"/(v['video_id']+'.npz')))
            with torch.no_grad():sim=rescore(m,v,route,clip,tokens)
            assert len(sim)==len(route['segments']) and np.isfinite(sim).all()
            del m,ds
    save(out/(args.stage+'.json'),dict(rows=reports,complete=True,seconds=time.time()-start,torch=torch.__version__,gpu_used=True))
    print('COMPLETED',args.stage,round(time.time()-start,2),flush=True)


if __name__=='__main__':main()
