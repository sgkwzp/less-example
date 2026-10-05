"""Oatmeal frozen-protocol support fit and full linear routes, CPU only."""
from pathlib import Path
import argparse,time
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import aem_cross_route_gpu as io
from actionness_gate import fit

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'hard_gate_transfer_runtime'

def register():
    data=ROOT/'code/AEM/data/egoper'
    jobs=[j for j in [__import__('json').loads(x) for x in (HERE/'g1_runtime/jobs.jsonl').read_text().splitlines()] if j['method']=='aem' and j['recipe']=='oatmeal' and j['K']=='2' and j['seed'] in [0,1]]
    ann={a['video_id']:a for a in io.read(data/'annotation.json')['oatmeal']['segments']}
    paths=[HERE/x for x in ['hard_gate_transfer.py','hard_gate_transfer_gpu.py','hard_gate_transfer_analyze.py','hard-gate-transfer-protocol.md',
        'actionness_gate.py','aem_cross_route_gpu.py','aem_scarcity_diagnostic.py','actionness_gate_analyze.py',
        'grr_validate.py','grr_smoke.py','fast_alignment.py','g1_runtime/jobs.jsonl','g1_runtime/oatmeal_aem_export.json',
        'aem_cross_route_runtime/gpu/registered.json','actionness_gate_runtime/results.json','actionness_gate_runtime/bootstrap_arrays.npz']]
    paths += [data/'annotation.json',data/'action2idx.json',ROOT/'code/GTG2Vid/dp/graph_utils.py']
    for j in jobs:
        assert len(j['test'])==35 and len(j['support_ids'])==2 and not set(j['test'])&set(j['support_ids'])
        assert all(not any(ann[v]['labels']['action_type']) for v in j['support_ids'])
        paths += [data/'oatmeal/vc_v_features_10fps'/(v+'.npy') for v in j['test']+j['support_ids']]
        paths += [HERE/'g1_runtime/aem_audit'/j['id']/(v+'.npz') for v in j['test']]
    assert not (OUT/'registered.json').exists()
    io.save(OUT/'registered.json',dict(created_unix=time.time(),jobs=jobs,support_annotations={v:ann[v] for j in jobs for v in j['support_ids']},
        hashes={p.relative_to(ROOT).as_posix():io.sha(p) for p in paths}))
    print('REGISTERED',io.sha(OUT/'registered.json'),flush=True)

def run():
    import torch,networkx as nx
    from grr_validate import Engine,segments
    torch.set_num_threads(1);reg=io.read(OUT/'registered.json')
    assert all(io.sha(ROOT/p)==h for p,h in reg['hashes'].items()) and not (OUT/'inputs_complete.json').exists()
    start=time.time();fits={};records=[]
    for j in reg['jobs']:
        seed=j['seed'];e=Engine.__new__(Engine);e.recipe='oatmeal';e.config=dict(top_m=5,similarity_logit_scale=8.);e.device='cpu'
        e.classes=16;e.drop_base=.3;e.node_base=0.;e.annotations={v:reg['support_annotations'][v] for v in j['support_ids']}
        e.tensors={};e.support_labels={};e.elapsed=dict(reference_seconds=0.,dp_seconds=0.,dtw_seconds=0.)
        xs=[e.features(v).numpy().astype(np.float64) for v in j['support_ids']];ys=[(e.labels(v)>0).astype(int) for v in j['support_ids']]
        folds=[];oof=[]
        for k,v in enumerate(j['support_ids']):
            th,info=fit([xs[1-k]],[ys[1-k]]);p=expit(xs[k]@th[:-1]+th[-1]);oof.append(p);y=ys[k]
            folds.append(dict(video=v,auc=float(roc_auc_score(y,p)),normal_action_rejection=float((p[y==1]<.5).mean()),
                background_retention=float((p[y==0]<.5).mean()),**info))
        theta,info=fit(xs,ys);yy=np.concatenate(ys);pp=np.concatenate(oof)
        fits[str(seed)]=dict(folds=folds,final=info,oof_auc=float(roc_auc_score(yy,pp)),oof_brier=float(np.mean((pp-yy)**2)))
        f=OUT/'fits'/f's{seed}.npz';f.parent.mkdir(exist_ok=True);assert not f.exists();np.savez_compressed(f,theta=theta,oof_y=yy,oof_p=pp)
        banks=e.banks(j['support_ids']);templates=[]
        counts=np.bincount(np.concatenate([e.labels(v) for v in j['support_ids']]),minlength=16);counts[0]=0;fallback=int(np.argmax(counts))
        for v in j['support_ids']:
            names=[f'{c},{i}' for i,(c,a,b) in enumerate(segments(e.labels(v))) if c>0]
            g=nx.DiGraph();g.add_nodes_from(names);g.add_edges_from(zip(names[:-1],names[1:]));templates.append((v,g,dict(enumerate(names))))
        for vid in j['test']:
            q=e.features(vid);sim=e.similarities(q,banks)
            (linear,dropped,cost),selected=min([(e.align(sim,g,ids),v) for v,g,ids in templates],key=lambda x:x[0][2])
            assert (linear>0).all() and (linear<16).all()
            p=expit(q.numpy().astype(np.float64)@theta[:-1]+theta[-1])
            dest=OUT/'inputs'/f's{seed}'/(vid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
            np.savez_compressed(dest,p=p,l_full_pred=linear,fallback=np.array(fallback))
            records.append(dict(seed=seed,video=vid,selected=selected,cost=cost,frames=len(p)));del e.tensors[vid]
        print('CPU_SEED_COMPLETE',seed,flush=True)
    paths=list((OUT/'inputs').glob('*/*.npz'))+list((OUT/'fits').glob('*.npz'))
    io.save(OUT/'inputs_complete.json',dict(complete=True,seconds=time.time()-start,fits=fits,records=records,
        registration_sha=io.sha(OUT/'registered.json'),hashes={p.relative_to(OUT).as_posix():io.sha(p) for p in paths}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['register','run']);a=p.parse_args();(register if a.stage=='register' else run)()
