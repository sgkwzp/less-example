"""Frozen AEM gap-only scores; no query annotations used for prediction."""
from pathlib import Path
import time
import numpy as np
import aem_cross_route_gpu as io

def main():
    root=Path('/hy-tmp');here=root/'ccfa-workfiles/experiments/fewshot-pmd';out=here/'actionness_gate_runtime'
    reg=io.read(out/'registered.json');inputs=io.read(out/'inputs_complete.json')
    assert inputs['complete'] and inputs['registration_sha']==io.sha(out/'registered.json')
    prior=io.read(here/'aem_cross_route_runtime/gpu/registered.json')
    assert all(io.sha(p)==h for p,h in prior['hashes'].items())
    for rel,h in reg['hashes'].items():
        if Path(rel).name in ['actionness_gate_gpu.py','actionness-gate-protocol.md','aem_cross_route_gpu.py'] or '/replay/C/' in rel or '/aem_support_route_runtime/scores/' in rel:
            assert io.sha(root/rel)==h,rel
    assert all(io.sha(out/p)==h for p,h in inputs['hashes'].items())
    assert not (out/'gpu_complete.json').exists()
    wall=time.time();torch,clip,tokens=io.initialize(root,out,1);start=time.time();checks=[]
    for j in reg['jobs']:
        m,ds=io.model_data(j,clip)
        for i in range(len(ds)):
            video=ds[i];vid=video['video_id'];seed=j['seed']
            # Remove labels before passing to feature/scoring functions.
            video={k:v for k,v in video.items() if k in ('video_id','feats','fps','duration','feat_stride','feat_num_frames')}
            inp=dict(np.load(out/'inputs'/f's{seed}'/(vid+'.npz')));bg=inp['bg'];n=len(bg);result={};errors={}
            with torch.no_grad():
                for name,path,key in [('N',here/'aem_cross_route_runtime/replay/C'/f's{seed}_KK','n'),
                                       ('L',here/'aem_support_route_runtime/scores'/f's{seed}','l')]:
                    z=np.load(path/(vid+'.npz'))
                    route={k:z[k] for k in ['segments','segment_action_pred']}
                    sim=io.rescore(m,video,route,clip,tokens)
                    err=float(np.abs(sim-z['segment_action_sim']).max());assert err<=1e-6,(vid,name,err)
                    errors[name]=err
                    ss=inp[key+'_gap_segments'];cc=inp[key+'_gap_classes']
                    gs=io.rescore(m,video,dict(segments=ss,segment_action_pred=cc),clip,tokens) if len(ss) else np.empty(0,np.float32)
                    full=z['scores'].copy();coverage=np.zeros(n,int)
                    for (a,b),s,c in zip(ss,gs,cc):
                        assert c>0 and bg[a:b+1].all();full[a:b+1]=1-float(s);coverage[a:b+1]+=1
                    assert np.array_equal(coverage,bg.astype(int))
                    assert np.array_equal(full[~bg],z['raw_all'][~bg])
                    assert np.array_equal(full*(~bg),z['scores'])
                    assert np.isfinite(full).all() and full.min()>=-1e-7 and full.max()<=1+1e-7
                    result['e_'+name]=full;result[key+'_gap_sim']=gs
                _,mask,points,logits,_=m.extract_feature([video])
                assert video['feats'].shape[-1]==n
                pt=points[0][:n].cpu().numpy();assert np.array_equal(pt[:,0],np.arange(n)) and (pt[:,-1]==1).all()
                valid=mask[0][0].cpu().numpy();assert valid[:n].all() and not valid[n:].any()
                prob=logits[0][0,:n].sigmoid().cpu().numpy();assert prob.shape==(n,16)
                result['aem_act']=prob[:,1:].max(1);result['aem_nonbg']=1-prob[:,0]
            dest=out/'scores'/f's{seed}'/(vid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
            np.savez_compressed(dest,**result)
            checks.append(dict(seed=seed,video=vid,control_max_difference=errors,n_frames=n))
        del m,ds;torch.cuda.empty_cache();print('GPU_SEED_COMPLETE',seed,flush=True)
    hashes={p.relative_to(out).as_posix():io.sha(p) for p in (out/'scores').glob('*/*.npz')}
    io.save(out/'gpu_complete.json',dict(complete=True,core_seconds=time.time()-start,wall_seconds=time.time()-wall,
        checks=checks,hashes=hashes,registration_sha=io.sha(out/'registered.json'),inputs_manifest_sha=io.sha(out/'inputs_complete.json'),torch=torch.__version__))
    print('GPU_COMPLETE',len(hashes),time.time()-wall,flush=True)

if __name__=='__main__':main()
