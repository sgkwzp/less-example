"""Registered K2 actionness experiment: support-only fit and query preparation."""
from pathlib import Path
import argparse
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import aem_cross_route_gpu as io

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'actionness_gate_runtime'

def register():
    prior=io.read(HERE/'aem_support_route_runtime/registered.json')
    paths=[HERE/p for p in ['actionness_gate.py','actionness_gate_gpu.py','actionness_gate_analyze.py',
        'actionness_history_audit.py','actionness-gate-protocol.md','actionness_gate_runtime/history_audit.json',
        'aem_support_route_runtime/registered.json','aem_support_route_runtime/routes_complete.json',
        'aem_support_route_runtime/gpu_complete.json','aem_cross_route_runtime/gpu/registered.json']]
    hashes=prior['hashes'].copy()
    for j in prior['jobs']:
        hist=HERE/'pilot_runtime/support_background'/f"coffee_k2_s{j['seed']}"
        paths += [hist/'manifest.json']
        for v in j['test']:
            paths += [hist/(v+'.npz'), HERE/'aem_support_route_runtime/scores'/f"s{j['seed']}"/(v+'.npz'),
                HERE/'aem_support_route_runtime/routes'/f"s{j['seed']}"/(v+'.npz')]
    hashes.update({p.relative_to(ROOT).as_posix():io.sha(p) for p in paths})
    assert all(io.sha(ROOT/p)==h for p,h in hashes.items())
    assert not (OUT/'registered.json').exists()
    io.save(OUT/'registered.json',dict(created_unix=time.time(),jobs=prior['jobs'],
        support_annotations=prior['support_annotations'],hashes=hashes))
    print('REGISTERED',io.sha(OUT/'registered.json'),flush=True)

def objective(theta,x,y,wt):
    z=x@theta[:-1]+theta[-1]
    loss=float(wt@(np.logaddexp(0,z)-y*z)+.0005*(theta[:-1]@theta[:-1]))
    r=wt*(expit(z)-y)
    return loss,np.r_[x.T@r+.001*theta[:-1],r.sum()]

def fit(xs,ys):
    x=np.vstack(xs).astype(np.float64);y=np.concatenate(ys).astype(np.float64)
    assert set(np.unique(y))=={0.,1.} and np.isfinite(x).all()
    wt=np.concatenate([np.full(len(a),1/(len(xs)*len(a))) for a in xs])
    result=minimize(objective,np.zeros(x.shape[1]+1),args=(x,y,wt),jac=True,method='L-BFGS-B',
                    options=dict(maxiter=1000,ftol=1e-12,gtol=1e-8))
    grad=float(np.max(np.abs(result.jac)))
    assert result.success and grad<1e-6,(result.message,grad)
    return result.x,dict(loss=float(result.fun),max_gradient=grad,iterations=result.nit,message=str(result.message))

def gap_segments(pred,bg):
    # Preserve every background gap independently, even with same neighboring class.
    from grr_validate import segments
    ss=[];cc=[]
    for isbg,a,b in segments(bg.astype(int)):
        if isbg:
            for c,lo,hi in segments(pred[a:b]):ss.append([a+lo,a+hi-1]);cc.append(c)
    return np.asarray(ss,dtype=np.int64).reshape(-1,2),np.asarray(cc,dtype=np.int64)

def nearest_route(native,fallback):
    ix=np.arange(len(native));pos=np.flatnonzero(native>0)
    if not len(pos):return np.full(len(native),fallback,np.int64),True
    right=np.clip(np.searchsorted(pos,ix),0,len(pos)-1);left=np.clip(np.searchsorted(pos,ix)-1,0,len(pos)-1)
    nearest=np.where(np.abs(pos[left]-ix)<=np.abs(pos[right]-ix),pos[left],pos[right])
    return native[nearest],False

def run():
    import torch
    import networkx as nx
    from grr_validate import Engine,segments
    torch.set_num_threads(1)
    reg=io.read(OUT/'registered.json');assert all(io.sha(ROOT/p)==h for p,h in reg['hashes'].items())
    assert not (OUT/'inputs_complete.json').exists()
    start=time.time();fits={};records=[]
    # One analytic-gradient check; no synthetic performance evidence.
    rng=np.random.default_rng(0);x=rng.normal(size=(9,4));y=np.arange(9)%2;w=np.full(9,1/9);th=rng.normal(size=5)
    _,grad=objective(th,x,y,w);num=np.array([(objective(th+np.eye(5)[i]*1e-5,x,y,w)[0]-objective(th-np.eye(5)[i]*1e-5,x,y,w)[0])/2e-5 for i in range(5)])
    gradient_error=float(np.max(np.abs(grad-num)));assert gradient_error<1e-8
    assert np.array_equal(nearest_route(np.array([0,2,0,0,0,3,0]),1)[0],np.array([2,2,2,2,3,3,3]))
    for j in reg['jobs']:
        seed=j['seed'];e=Engine.__new__(Engine)
        e.recipe='coffee';e.config=dict(top_m=5,similarity_logit_scale=8.);e.device='cpu'
        e.classes=16;e.drop_base=.3;e.node_base=0.
        e.annotations={v:reg['support_annotations'][v] for v in j['support_ids']}
        e.tensors={};e.support_labels={};e.elapsed=dict(reference_seconds=0.,dp_seconds=0.,dtw_seconds=0.)
        xs=[e.features(v).numpy().astype(np.float64) for v in j['support_ids']]
        ys=[(e.labels(v)>0).astype(int) for v in j['support_ids']]
        folds=[];oof=[]
        for k,v in enumerate(j['support_ids']):
            theta,info=fit([xs[1-k]],[ys[1-k]]);p=expit(xs[k]@theta[:-1]+theta[-1]);oof.append(p)
            folds.append(dict(video=v,auc=float(roc_auc_score(ys[k],p)),**info))
        theta,info=fit(xs,ys)
        yy=np.concatenate(ys);pp=np.concatenate(oof)
        fits[str(seed)]=dict(support_ids=j['support_ids'],folds=folds,final=info,
            oof_auc=float(roc_auc_score(yy,pp)),oof_brier=float(np.mean((yy-pp)**2)),
            positive_fractions=[float(y.mean()) for y in ys])
        dest=OUT/'fits'/f's{seed}.npz';dest.parent.mkdir(exist_ok=True)
        assert not dest.exists();np.savez_compressed(dest,theta=theta,oof_p=pp,oof_y=yy)
        banks=e.banks(j['support_ids']);templates=[]
        allcls=np.concatenate([e.labels(v) for v in j['support_ids']]);counts=np.bincount(allcls,minlength=16);counts[0]=0;fallback=int(np.argmax(counts))
        for v in j['support_ids']:
            names=[f'{c},{i}' for i,(c,a,b) in enumerate(segments(e.labels(v))) if c>0]
            g=nx.DiGraph();g.add_nodes_from(names);g.add_edges_from(zip(names[:-1],names[1:]))
            templates.append((v,g,dict(enumerate(names))))
        histdir=HERE/'pilot_runtime/support_background'/f'coffee_k2_s{seed}'
        hist=io.read(histdir/'manifest.json');assert hist['support_ids']==j['support_ids'] and hist['test_ids']==j['test']
        rule=hist['rule']
        for vid in j['test']:
            q=e.features(vid);sim=e.similarities(q,banks)
            choices=[(e.align(sim,g,ids),v) for v,g,ids in templates]
            (linear,dropped,cost),selected=min(choices,key=lambda a:a[0][2])
            native=np.load(HERE/'aem_cross_route_runtime/replay/C'/f's{seed}_KK'/(vid+'.npz'))
            old=np.load(HERE/'aem_support_route_runtime/routes'/f's{seed}'/(vid+'.npz'))
            bg=native['pred_action']==0
            assert selected==str(old['selected_support']) and abs(cost-float(old['alignment_cost']))<1e-10
            assert (linear>0).all() and np.array_equal(np.where(bg,0,linear),old['pred_action'])
            nroute,used=nearest_route(native['pred_action'],fallback)
            ns,nc=gap_segments(nroute,bg);ls,lc=gap_segments(linear,bg)
            hb=(sim[:,0]-sim[:,1:].max(1)>rule['margin_threshold'])&(1-sim[:,0]<=rule['background_cost_max']) if rule['enabled'] else np.zeros(len(q),bool)
            assert np.array_equal(hb,np.load(histdir/(vid+'.npz'))['pred_action']==0)
            p=expit(q.numpy().astype(np.float64)@theta[:-1]+theta[-1])
            dest=OUT/'inputs'/f's{seed}'/(vid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
            np.savez_compressed(dest,p=p,bg=bg,hist_bg=hb,n_gap_segments=ns,n_gap_classes=nc,
                l_gap_segments=ls,l_gap_classes=lc,n_full_pred=nroute,l_full_pred=linear)
            records.append(dict(seed=seed,video=vid,selected=selected,cost=cost,n_gap_segments=len(ns),l_gap_segments=len(ls),all_background_fallback=used))
            del e.tensors[vid]
        print('CPU_SEED_COMPLETE',seed,flush=True)
    paths=list((OUT/'inputs').glob('*/*.npz'))+list((OUT/'fits').glob('*.npz'))
    io.save(OUT/'inputs_complete.json',dict(complete=True,seconds=time.time()-start,fits=fits,records=records,
        gradient_error=gradient_error,registration_sha=io.sha(OUT/'registered.json'),
        hashes={p.relative_to(OUT).as_posix():io.sha(p) for p in paths}))
    print('CPU_COMPLETE',time.time()-start,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['register','run']);args=parser.parse_args()
    (register if args.stage=='register' else run)()
