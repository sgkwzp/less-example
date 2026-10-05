"""K-only linear routing, preserving the frozen AEM background mask."""
from pathlib import Path
import argparse
import time
import numpy as np
import aem_cross_route_gpu as io

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'aem_support_route_runtime'

def prepare():
    from grr_smoke import DATA, FEATURE_SUBDIR
    jobs=[j for j in io.read(HERE/'aem_cross_route_runtime/gpu/registered.json')['jobs'] if j['K']=='2']
    assert len(jobs)==2 and {j['seed'] for j in jobs}=={0,1}
    ann={v['video_id']:v for v in io.read(DATA/'annotation.json')['coffee']['segments']}
    paths=[HERE/f for f in ['aem_support_route.py','aem_support_route_gpu.py','aem_support_route_analyze.py',
        'aem-support-route-protocol.md','grr_validate.py','grr_smoke.py','fast_alignment.py',
        'aem_scarcity_diagnostic.py','aem_cross_route_gpu.py','aem_cross_route_runtime/gpu/registered.json']]
    paths += [ROOT/'code/GTG2Vid/dp/graph_utils.py',DATA/'annotation.json',HERE/'g1_runtime/jobs.jsonl']
    for j in jobs:
        assert len(j['support_ids'])==2 and not set(j['support_ids'])&set(j['test'])
        assert all(not any(ann[v]['labels']['action_type']) for v in j['support_ids'])
        paths += [DATA/'coffee'/FEATURE_SUBDIR/(v+'.npy') for v in j['support_ids']+j['test']]
        paths += [HERE/'aem_cross_route_runtime/replay/C'/f"s{j['seed']}_KK"/(v+'.npz') for v in j['test']]
    OUT.mkdir(exist_ok=True);assert not (OUT/'registered.json').exists()
    io.save(OUT/'registered.json',dict(created_unix=time.time(),jobs=jobs,
        support_annotations={v:ann[v] for j in jobs for v in j['support_ids']},
        hashes={p.relative_to(ROOT).as_posix():io.sha(p) for p in sorted(set(paths))}))
    print('REGISTERED',io.sha(OUT/'registered.json'),flush=True)

def routes():
    import torch
    import networkx as nx
    from grr_validate import Engine,segments
    torch.set_num_threads(1)
    reg=io.read(OUT/'registered.json')
    assert all(io.sha(ROOT/p)==h for p,h in reg['hashes'].items())
    start=time.time();records=[];smoke=None
    for j in reg['jobs']:
        # Deliberately bypass Engine.__init__: it loads the released task DAG.
        e=Engine.__new__(Engine)
        e.recipe='coffee';e.config=dict(top_m=5,similarity_logit_scale=8.);e.device='cpu'
        e.classes=16;e.drop_base=.3;e.node_base=0.
        e.annotations={v:reg['support_annotations'][v] for v in j['support_ids']}
        e.tensors={};e.support_labels={};e.elapsed=dict(reference_seconds=0.,dp_seconds=0.,dtw_seconds=0.)
        banks=e.banks(j['support_ids']);templates=[]
        for v in j['support_ids']:
            names=[f'{c},{i}' for i,(c,a,b) in enumerate(segments(e.labels(v))) if c>0]
            g=nx.DiGraph();g.add_nodes_from(names);g.add_edges_from(zip(names[:-1],names[1:]))
            templates.append((v,g,dict(enumerate(names))))
        for vid in j['test']:
            q=e.features(vid);sim=e.similarities(q,banks)
            if smoke is None:
                assert '_normal_' in vid
                ix=np.linspace(0,len(q)-1,11,dtype=int);err=0.
                for c,bank in banks.items():
                    brute=q.numpy()[ix]@bank.numpy().T;k=min(5,len(bank))
                    ref=np.sort(brute,axis=1)[:,-k:].mean(1)
                    err=max(err,float(np.abs(ref-sim[ix,c]).max()))
                assert err<1e-6;smoke=dict(video=vid,independent_top5_error=err)
            choices=[(e.align(sim,g,ids),v) for v,g,ids in templates]
            (linear,dropped,cost),selected=min(choices,key=lambda x:x[0][2])
            native=np.load(HERE/'aem_cross_route_runtime/replay/C'/f"s{j['seed']}_KK"/(vid+'.npz'))
            bg=native['pred_action']==0
            assert len(bg)==len(linear) and (linear>0).all() and (linear<16).all()
            pred=np.where(bg,0,linear).astype(np.int64)
            runs=segments(pred);seg=np.array([[a,b-1] for c,a,b in runs],np.int64);cls=np.array([c for c,a,b in runs],np.int64)
            assert np.array_equal(np.repeat(cls,seg[:,1]-seg[:,0]+1),pred)
            dest=OUT/'routes'/f"s{j['seed']}"/(vid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
            assert not dest.exists()
            np.savez_compressed(dest,segments=seg,segment_action_pred=cls,pred_action=pred,
                selected_support=np.array(selected),alignment_cost=np.array(cost))
            records.append(dict(seed=j['seed'],video=vid,selected=selected,cost=cost,
                old_segments=len(native['segments']),new_segments=len(seg),singletons=int((seg[:,0]==seg[:,1]).sum())))
            del e.tensors[vid]
    hashes={p.relative_to(OUT).as_posix():io.sha(p) for p in sorted((OUT/'routes').glob('*/*.npz'))}
    io.save(OUT/'routes_complete.json',dict(complete=True,seconds=time.time()-start,smoke=smoke,records=records,hashes=hashes))
    print('ROUTES_COMPLETE',len(hashes),'seconds',time.time()-start,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','routes']);a=p.parse_args()
    (prepare if a.stage=='prepare' else routes)()
