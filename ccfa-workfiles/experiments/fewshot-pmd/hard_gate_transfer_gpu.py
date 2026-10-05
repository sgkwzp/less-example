"""Oatmeal native reproduction and frozen segment scoring, no query GT prediction."""
from pathlib import Path
import time
import numpy as np
import aem_cross_route_gpu as io

def runs(pred):
    edges=np.r_[0,np.flatnonzero(pred[1:]!=pred[:-1])+1,len(pred)]
    return np.array([[a,b-1] for a,b in zip(edges[:-1],edges[1:])],np.int64),pred[edges[:-1]].astype(np.int64)

def gap_runs(pred,bg):
    ss=[];cc=[]
    for (a,b),c in zip(*runs(bg.astype(int))):
        if c:
            seg,cls=runs(pred[a:b+1]);ss.extend((seg+a).tolist());cc.extend(cls.tolist())
    return np.asarray(ss,np.int64).reshape(-1,2),np.asarray(cc,np.int64)

def nroute(pred,fallback):
    pos=np.flatnonzero(pred>0);ix=np.arange(len(pred))
    if not len(pos):return np.full(len(pred),fallback,np.int64),True
    k=np.searchsorted(pos,ix);left=pos[np.clip(k-1,0,len(pos)-1)];right=pos[np.clip(k,0,len(pos)-1)]
    return pred[np.where(abs(ix-left)<=abs(ix-right),left,right)],False

def expand(seg,sim,n):
    out=np.zeros(n,np.float32)
    for (a,b),s in zip(seg,sim):out[a:b+1]=1-float(s)
    return out

def main():
    root=Path('/hy-tmp');here=root/'ccfa-workfiles/experiments/fewshot-pmd';out=here/'hard_gate_transfer_runtime'
    reg=io.read(out/'registered.json');cpu=io.read(out/'inputs_complete.json');assert cpu['complete']
    assert cpu['registration_sha']==io.sha(out/'registered.json')
    assert all(io.sha(out/p)==h for p,h in cpu['hashes'].items())
    prior=io.read(here/'aem_cross_route_runtime/gpu/registered.json');assert all(io.sha(p)==h for p,h in prior['hashes'].items())
    for p,h in reg['hashes'].items():
        if Path(p).name in ['hard_gate_transfer_gpu.py','hard-gate-transfer-protocol.md','aem_cross_route_gpu.py','annotation.json'] or '/g1_runtime/aem_audit/' in p or '/vc_v_features_10fps/' in p:
            assert io.sha(root/p)==h,p
    historical={x['id']:x for x in io.read(here/'g1_runtime/oatmeal_aem_export.json')['rows']}
    hashes={}
    import yaml
    for j in reg['jobs']:
        assert io.sha(j['checkpoint'])==historical[j['id']]['checkpoint_sha256']
        assert io.sha(j['config'])==historical[j['id']]['config_sha256']
        cfg=yaml.safe_load(Path(j['config']).read_text());sf=Path(cfg['dataset']['support_file']);assert sf.read_text().split()==j['support_ids']
        for p in [Path(j['checkpoint']),Path(j['config']),sf]:hashes[str(p)]=io.sha(p)
    assert not (out/'gpu_registered.json').exists();io.save(out/'gpu_registered.json',dict(created_unix=time.time(),hashes=hashes,prior_identity_sha=io.sha(here/'aem_cross_route_runtime/gpu/registered.json')))
    wall=time.time();torch,clip,_=io.initialize(root,out,1)
    from libs.datasets.clip_encode import action_tokenize_egoper
    tokens=action_tokenize_egoper(clip,'oatmeal',io.read(root/'code/AEM/data/egoper/annotation.json')['oatmeal'],device='cuda:0')
    start=time.time();checks=[]
    for j in reg['jobs']:
        m,ds=io.model_data(j,clip)
        for i in range(len(ds)):
            item=ds[i];vid=item['video_id'];s=j['seed'];inp=np.load(out/'inputs'/f's{s}'/(vid+'.npz'));n=len(inp['p'])
            video={k:v for k,v in item.items() if k in ['video_id','feats','fps','duration','feat_stride','feat_num_frames']}
            # Original postprocessing expects annotation-shaped metadata; give a single dummy normal segment.
            video.update(segments=torch.tensor([[0,n]],dtype=torch.float32),labels=torch.zeros(1,dtype=torch.long),labels_error=torch.zeros(1,dtype=torch.long))
            assert video['feats'].shape[-1]==n
            old=np.load(here/'g1_runtime/aem_audit'/j['id']/(vid+'.npz'))
            with torch.no_grad():
                oldsim=io.rescore(m,video,{k:old[k] for k in ['segments','segment_action_pred']},clip,tokens)
                olderr=float(abs(oldsim-old['segment_action_sim']).max());assert olderr<=1e-6,(vid,olderr)
                native=m.inference([video],clip,tokens)[vid]
                seg=native['segments'].numpy().astype(np.int64);cls=native['segment_action_pred'].numpy().astype(np.int64)
                pred=native['frame_action_pred'].numpy()[:n].astype(np.int64);sim=native['segment_action_sim'].cpu().numpy().reshape(-1)
                assert np.array_equal(np.repeat(cls,seg[:,1]-seg[:,0]+1),pred)
                replay=io.rescore(m,video,dict(segments=seg,segment_action_pred=cls),clip,tokens)
                err=float(abs(replay-sim).max());assert err<=1e-6
                bg=pred==0;nk=expand(seg,sim,n)*(~bg);linear=inp['l_full_pred'];lpred=np.where(bg,0,linear)
                ls,lc=runs(lpred);lsim=io.rescore(m,video,dict(segments=ls,segment_action_pred=lc),clip,tokens)
                lk=expand(ls,lsim,n)*(~bg);fulln,fb=nroute(pred,int(inp['fallback']))
                output=dict(native_pred=pred,native_segments=seg,native_classes=cls,native_sim=sim,
                    l_segments=ls,l_classes=lc,l_sim=lsim,NK=nk,LK=lk)
                for key,route,base in [('N',fulln,nk),('L',linear,lk)]:
                    gs,gc=gap_runs(route,bg);gsi=io.rescore(m,video,dict(segments=gs,segment_action_pred=gc),clip,tokens) if len(gs) else np.empty(0,np.float32)
                    e=base.copy();coverage=np.zeros(n,int)
                    for (a,b),v in zip(gs,gsi):e[a:b+1]=1-float(v);coverage[a:b+1]+=1
                    assert np.array_equal(coverage,bg.astype(int)) and np.array_equal(e*(~bg),base)
                    assert np.isfinite(e).all() and e.min()>=0 and e.max()<=1
                    output.update({key+'_gap_segments':gs,key+'_gap_classes':gc,key+'_gap_sim':gsi,'e_'+key:e})
            dest=out/'scores'/f's{s}'/(vid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
            np.savez_compressed(dest,**output)
            checks.append(dict(seed=s,video=vid,historical_fixed_route_error=olderr,native_replay_error=err,
                historical_route_changed_frames=int((pred!=old['pred_action']).sum()),historical_score_max_difference=float(abs(nk-old['scores']).max()),
                all_background_fallback=fb))
        del m,ds;torch.cuda.empty_cache();print('GPU_SEED_COMPLETE',s,flush=True)
    paths=list((out/'scores').glob('*/*.npz'));assert len(paths)==70
    io.save(out/'gpu_complete.json',dict(complete=True,core_seconds=time.time()-start,wall_seconds=time.time()-wall,
        checks=checks,registration_sha=io.sha(out/'registered.json'),inputs_sha=io.sha(out/'inputs_complete.json'),
        gpu_registration_sha=io.sha(out/'gpu_registered.json'),hashes={p.relative_to(out).as_posix():io.sha(p) for p in paths}))
    print('GPU_COMPLETE',len(paths),time.time()-wall,flush=True)

if __name__=='__main__':main()
