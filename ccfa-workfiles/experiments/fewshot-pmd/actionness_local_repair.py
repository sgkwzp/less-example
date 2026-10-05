"""CPU cached 2x2 intervention: retain action scores and recover background gaps."""
from pathlib import Path
import argparse
import time
import numpy as np
from scipy.ndimage import median_filter,maximum_filter1d
import aem_cross_route_gpu as io
from actionness_gate_analyze import load_rows,explicit
from aem_scarcity_diagnostic import metrics_boot,contrast,decomposition,auc

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
PRIOR=HERE/'actionness_gate_runtime';OUT=HERE/'actionness_local_repair_runtime'
METHODS=['NK','LK','L_FG','L_GAP','L_P','E_L','L_HARD']
OLD=['NK','LK','L_P','E_L','L_HARD']
PAIRS=[('L_GAP','LK'),('L_GAP','NK'),('L_GAP','L_HARD'),('L_GAP','E_L'),
       ('L_FG','LK'),('L_P','L_GAP'),('L_P','L_FG')]

def register():
    reg=io.read(PRIOR/'registered.json');hashes=reg['hashes'].copy()
    for name in ['inputs_complete.json','gpu_complete.json']:
        for p,h in io.read(PRIOR/name)['hashes'].items():hashes[(PRIOR/p).relative_to(ROOT).as_posix()]=h
    paths=[Path(__file__),HERE/'actionness-local-repair-protocol.md',HERE/'actionness_gate_analyze.py',
           HERE/'aem_scarcity_diagnostic.py']+[PRIOR/p for p in ['registered.json','inputs_complete.json',
           'gpu_complete.json','results.json','verification.json','bootstrap_arrays.npz']]
    hashes.update({p.relative_to(ROOT).as_posix():io.sha(p) for p in paths})
    assert all(io.sha(ROOT/p)==h for p,h in hashes.items())
    assert not (OUT/'registered.json').exists()
    io.save(OUT/'registered.json',dict(created_unix=time.time(),jobs=reg['jobs'],hashes=hashes,
        prior_result_sha=io.sha(PRIOR/'results.json'),scope='Post-result development, frozen before new score combinations'))
    print('REGISTERED',io.sha(OUT/'registered.json'),flush=True)

def mask_pairs(rows,m):
    y=np.concatenate([r['y'] for r in rows]);bg=np.concatenate([r['bg'] for r in rows]);s=np.concatenate([r[m] for r in rows])
    den=int(y.sum())*int((~y).sum());out={}
    for ep in [False,True]:
        p=s[y&(bg==ep)]
        for nq in [False,True]:
            q=np.sort(s[(~y)&(bg==nq)]);n=len(p)*len(q)
            wins=float(((np.searchsorted(q,p,'left')+np.searchsorted(q,p,'right'))*.5).sum())
            name=('rejected' if ep else 'retained')+'_error_vs_'+('rejected' if nq else 'retained')+'_normal'
            out[name]=dict(pairs=n,contribution=wins/den,auc=wins/n if n else None)
    assert abs(sum(v['contribution'] for v in out.values())-auc(y,s))<1e-12
    return out

def run():
    start=time.time();reg=io.read(OUT/'registered.json');assert not (OUT/'results.json').exists()
    assert all(io.sha(ROOT/p)==h for p,h in reg['hashes'].items())
    ann={a['video_id']:a['labels'] for a in io.read(ROOT/'code/AEM/data/egoper/annotation.json')['coffee']['segments']}
    old=io.read(PRIOR/'results.json');weights=np.load(PRIOR/'bootstrap_arrays.npz')['weights']
    expected=np.vstack([np.ones(38,int),np.random.default_rng(20261005).multinomial(38,np.full(38,1/38),size=2000)])
    assert np.array_equal(weights,expected)
    rows_by_seed={};checks=[];regions={};maxerr=0.;saved={}
    for j in reg['jobs']:
        seed=str(j['seed']);rows=load_rows(j,ann);assert len(rows)==38
        for r in rows:
            bg=r['bg'];r['L_FG']=np.where(bg,0.,r['L_P']);r['L_GAP']=np.where(bg,r['L_P'],r['E_L'])
            assert np.array_equal(r['L_GAP'][~bg],r['LK'][~bg])
            assert np.array_equal(r['L_GAP'][bg],r['L_P'][bg]) and (r['L_FG'][bg]==0).all()
            assert np.allclose(r['L_FG']+r['L_GAP']-r['LK'],r['L_P'],atol=1e-15,rtol=0)
            stable=(maximum_filter1d(bg.astype(int),21,mode='nearest')==0)
            diff=np.abs(median_filter(r['L_GAP'],21,mode='nearest')-median_filter(r['LK'],21,mode='nearest'))
            assert (diff[stable]==0).all()
            checks.append(dict(seed=int(seed),video=r['video'],stable_retained_frames=int(stable.sum()),
                smoothing_changed_retained_frames=int(((diff>0)&~bg).sum())))
            dest=OUT/'scores'/f's{seed}'/(r['video']+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
            np.savez_compressed(dest,L_FG_raw=r['L_FG'],L_GAP_raw=r['L_GAP'])
        rows_by_seed[seed]=rows
        region={}
        for name,isbg in [('retained',False),('rejected',True)]:
            subset=[]
            for r in rows:
                k=r['bg']==isbg
                subset.append({key:r[key][k] for key in ['gt','y','types','E_L','P','L_P']})
            metric,ar,err=metrics_boot(subset,['E_L','P','L_P'],weights);maxerr=max(maxerr,err)
            gt=np.concatenate([r['gt'] for r in subset]);y=np.concatenate([r['y'] for r in subset]);mask=((gt>0)&y)|((gt==0)&~y)
            region[name]=dict(raw_metrics=metric,soft_minus_residual=contrast(ar,'L_P','E_L'),
                error_vs_true_background_auc={m:auc(y[mask],np.concatenate([r[m] for r in subset])[mask]) for m in ['E_L','P','L_P']})
        regions[seed]=region
    result={}
    for mode in ['raw','median21']:
        outputs={};bundles=[]
        for seed,raw in rows_by_seed.items():
            rows=[{**r,**{m:median_filter(r[m],21,mode='nearest') if mode=='median21' else r[m] for m in METHODS}} for r in raw]
            metrics,arr,err=metrics_boot(rows,METHODS,weights);maxerr=max(maxerr,err);bundles.append(arr)
            for m in OLD:
                for k in ['whole','known','within']:
                    err=abs(metrics[m][k]['auc']-old['analyses'][mode]['seeds'][seed]['metrics'][m][k]['auc']);assert err<1e-12;maxerr=max(maxerr,err)
            for m in ['L_FG','L_GAP']:
                for k in ['whole','known','within']:
                    for draw in [0,1]:
                        err=abs(explicit(rows,weights[draw],m,k)-arr[m][k][draw]);assert err<1e-12;maxerr=max(maxerr,err)
            gt=np.concatenate([r['gt'] for r in rows]);y=np.concatenate([r['y'] for r in rows])
            assert auc(y,np.full(len(y),.5))==.5
            dec={m:decomposition(gt,y,np.concatenate([r[m] for r in rows])) for m in METHODS}
            masks={m:mask_pairs(rows,m) for m in METHODS}
            pd={a+'__minus__'+b:{k:100*(dec[a]['groups'][k]['contribution']-dec[b]['groups'][k]['contribution']) for k in dec[a]['groups']} for a,b in PAIRS}
            md={a+'__minus__'+b:{k:100*(masks[a][k]['contribution']-masks[b][k]['contribution']) for k in masks[a]} for a,b in PAIRS}
            outputs[seed]=dict(metrics=metrics,contrasts={a+'__minus__'+b:contrast(arr,a,b) for a,b in PAIRS},
                gt_pair_differences_pp=pd,mask_pair_differences_pp=md,gt_decomposition=dec,mask_decomposition=masks)
            for m in METHODS:
                for k in arr[m]:saved[f'{mode}_s{seed}_{m}_{k}']=arr[m][k]
            print('ANALYZED',mode,seed,flush=True)
        mean={m:{k:np.mean([a[m][k] for a in bundles],axis=0) for k in bundles[0][m]} for m in METHODS}
        interaction={k:mean['L_P'][k]-mean['L_FG'][k]-mean['L_GAP'][k]+mean['LK'][k] for k in mean['LK']}
        inter=contrast({'interaction':interaction,'zero':{k:np.zeros_like(v) for k,v in interaction.items()}},'interaction','zero')
        result[mode]=dict(seeds=outputs,mean_auc={m:{k:float(v[0]) for k,v in d.items()} for m,d in mean.items()},
            mean_contrasts={a+'__minus__'+b:contrast(mean,a,b) for a,b in PAIRS},interaction=inter)
        for field in ['gt_pair_differences_pp','mask_pair_differences_pp']:
            result[mode]['mean_'+field]={a+'__minus__'+b:{k:float(np.mean([v[field][a+'__minus__'+b][k] for v in outputs.values()])) for k in next(iter(outputs.values()))[field][a+'__minus__'+b]} for a,b in PAIRS}
    co=result['median21']['mean_contrasts']
    gain=all(co['L_GAP__minus__'+b]['whole']['ci95_pp'][0]>0 for b in ['NK','LK'])
    protect=all(co['L_GAP__minus__'+b][k]['delta_pp']>=0 for b in ['NK','LK'] for k in ['known','within'])
    decision=dict(overall_gain_pass=gain,development_point_protection_pass=protect,extension_pass=gain and protect)
    io.save(OUT/'results.json',dict(analyses=result,regions=regions,decision=decision))
    np.savez_compressed(OUT/'bootstrap_arrays.npz',weights=weights,**saved)
    hashes={p.relative_to(OUT).as_posix():io.sha(p) for p in (OUT/'scores').glob('*/*.npz')}
    assert len(hashes)==76 and all(io.sha(ROOT/p)==h for p,h in reg['hashes'].items())
    io.save(OUT/'verification.json',dict(complete=True,gpu_used=False,seconds=time.time()-start,
        all_hashes_unchanged=True,independent_and_old_reproduction_max_error=maxerr,constant_auc=.5,
        region_invariance_checks=checks,score_hashes=hashes,registration_sha=io.sha(OUT/'registered.json'),
        result_sha=io.sha(OUT/'results.json'),bootstrap_sha=io.sha(OUT/'bootstrap_arrays.npz')))
    print('DECISION',decision,flush=True)
    for m,d in result['median21']['mean_auc'].items():print(m,{k:round(v*100,4) for k,v in d.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['register','run']);args=p.parse_args()
    (register if args.stage=='register' else run)()
