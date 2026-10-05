"""Frozen hard-gate transfer evaluation, annotations read only here."""
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import median_filter
import aem_cross_route_gpu as io
from aem_scarcity_diagnostic import metrics_boot,contrast,decomposition,annotation_frames,auc,kernel,evaluate,independent_auc
from actionness_gate_analyze import explicit

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'hard_gate_transfer_runtime'
METHODS=['NK','LK','E_L','L_GAP','N_HARD','L_HARD','L_HARD_04','L_HARD_06']
PAIRS=[('L_HARD',b) for b in ['NK','LK','E_L','L_GAP','N_HARD']]

def shared_mean(bundles):
    return {m:{k:np.mean([a[m][k] for a in bundles],axis=0) for k in bundles[0][m]} for m in METHODS}

def joint(bundles,seed):
    choices=np.random.default_rng(seed).integers(0,2,size=(2000,2));draw=np.arange(1,2001)[:,None]
    return {m:{k:np.r_[np.mean([a[m][k][0] for a in bundles]),np.stack([a[m][k] for a in bundles])[choices,draw].mean(1)]
        for k in bundles[0][m]} for m in bundles[0]}

def load(j,ann):
    rows=[]
    for vid in j['test']:
        seed=j['seed'];x=np.load(OUT/'inputs'/f's{seed}'/(vid+'.npz'));z=np.load(OUT/'scores'/f's{seed}'/(vid+'.npz'))
        p=x['p'];n=len(p);gt,typ=annotation_frames(ann[vid],n);old=np.load(HERE/'g1_runtime/aem_audit'/j['id']/(vid+'.npz'))
        assert np.array_equal(gt,old['gt_action']) and np.array_equal(typ>0,old['labels']>0)
        bg=z['native_pred']==0;el=z['e_L'];en=z['e_N'];assert np.array_equal(el*(~bg),z['LK']) and np.array_equal(en*(~bg),z['NK'])
        for k,base in [('N',z['NK']),('L',z['LK'])]:
            re=base.copy();cover=np.zeros(n,int)
            for (a,b),c,s in zip(z[k+'_gap_segments'],z[k+'_gap_classes'],z[k+'_gap_sim']):
                assert c>0 and bg[a:b+1].all();re[a:b+1]=1-float(s);cover[a:b+1]+=1
            assert np.array_equal(re,z['e_'+k]) and np.array_equal(cover,bg.astype(int))
        gaplen=np.zeros(n,int)
        for a,b in z['L_gap_segments']:gaplen[a:b+1]=b-a+1
        rows.append(dict(video=vid,gt=gt,types=typ,y=typ>0,p=p,bg=bg,l_pred=x['l_full_pred'],gaplen=gaplen,
            NK=z['NK'],LK=z['LK'],E_L=el,L_GAP=np.where(bg,p*el,el),N_HARD=(p>=.5)*en,
            L_HARD=(p>=.5)*el,L_HARD_04=(p>=.4)*el,L_HARD_06=(p>=.6)*el))
    return rows

def ratios(rows):
    gt=np.concatenate([r['gt'] for r in rows]);y=np.concatenate([r['y'] for r in rows]);p=np.concatenate([r['p'] for r in rows]);bg=np.concatenate([r['bg'] for r in rows])
    masks=dict(addition=(gt<0)&y,known_error=(gt>0)&y,normal_action=(gt>0)&~y,true_background=(gt==0)&~y)
    out={}
    for label,mask in masks.items():
        out[label]=dict(frames=int(mask.sum()),native_rejection=float(bg[mask].mean()),probe_rejection=float((p[mask]<.5).mean()))
    out['addition_minus_normal_action_rejection']={k:out['addition'][k]-out['normal_action'][k] for k in ['native_rejection','probe_rejection']}
    return out

def diagnostics(rows):
    gt=np.concatenate([r['gt'] for r in rows]);y=np.concatenate([r['y'] for r in rows]);bg=np.concatenate([r['bg'] for r in rows]);pred=np.concatenate([r['l_pred'] for r in rows]);e=np.concatenate([r['E_L'] for r in rows]);length=np.concatenate([r['gaplen'] for r in rows])
    accuracy={};normal={}
    for region,b in [('retained',False),('rejected',True)]:
        for label,iserr in [('normal',False),('error',True)]:
            mask=(bg==b)&(gt>0)&(y==iserr)
            accuracy[region+'_'+label]=dict(frames=int(mask.sum()),accuracy=float((pred[mask]==gt[mask]).mean()) if mask.any() else None)
        mask=(bg==b)&(gt>0)&~y
        normal[region]=dict(frames=int(mask.sum()),mean=float(e[mask].mean()),quantiles=np.quantile(e[mask],[.1,.5,.9]).tolist())
    paired=[]
    for r in rows:
        for step in np.unique(r['gt'][r['gt']>0]):
            ok=(r['gt']==step)&~r['y'];a=ok&r['bg'];b=ok&~r['bg']
            if a.any() and b.any():paired.append(dict(video=r['video'],step=int(step),rejected_frames=int(a.sum()),retained_frames=int(b.sum()),mean_difference=float(r['E_L'][a].mean()-r['E_L'][b].mean())))
    bins={}
    for lo,hi in [(1,10),(11,30),(31,100),(101,10**9)]:
        mask=bg&(length>=lo)&(length<=hi)&(gt>0)
        bins[f'{lo}-{hi}']=dict(frames=int(mask.sum()),errors=int(y[mask].sum()),normal=int((~y[mask]).sum()),
            error_auc=auc(y[mask],e[mask]),route_accuracy=float((pred[mask]==gt[mask]).mean()) if mask.any() else None,
            by_step={str(int(step)):auc(y[mask&(gt==step)],e[mask&(gt==step)]) for step in np.unique(gt[mask])})
    return dict(route_accuracy=accuracy,normal_residual=normal,matched_video_step_normal_differences=paired,
        matched_equal_cell_mean_difference=float(np.mean([v['mean_difference'] for v in paired])) if paired else None,gap_length_bins=bins)

def main():
    start=time.time();reg=io.read(OUT/'registered.json');cpu=io.read(OUT/'inputs_complete.json');gpu=io.read(OUT/'gpu_complete.json')
    assert not (OUT/'results.json').exists()
    assert cpu['complete'] and gpu['complete'] and gpu['registration_sha']==io.sha(OUT/'registered.json') and gpu['inputs_sha']==io.sha(OUT/'inputs_complete.json')
    assert gpu['gpu_registration_sha']==io.sha(OUT/'gpu_registered.json')
    assert all(io.sha(ROOT/p)==h for p,h in reg['hashes'].items())
    assert all(io.sha(OUT/p)==h for d in [cpu,gpu] for p,h in d['hashes'].items())
    ann={a['video_id']:a['labels'] for a in io.read(ROOT/'code/AEM/data/egoper/annotation.json')['oatmeal']['segments']}
    jobs=reg['jobs'];assert len(jobs)==2 and jobs[0]['test']==jobs[1]['test'] and len(jobs[0]['test'])==35
    weights=np.vstack([np.ones(35,int),np.random.default_rng(20261006).multinomial(35,np.full(35,1/35),size=2000)])
    raw={str(j['seed']):load(j,ann) for j in jobs};analyses={};saved={};maxerr=0.;mechanism={}
    for s,rows in raw.items():mechanism[s]=dict(rejection=ratios(rows),diagnostics=diagnostics(rows))
    for mode in ['raw','median21']:
        output={};bundles=[]
        for s,rr in raw.items():
            rows=[{**r,**{m:median_filter(r[m],21,mode='nearest') if mode=='median21' else r[m] for m in METHODS}} for r in rr]
            metric,arr,err=metrics_boot(rows,METHODS,weights);maxerr=max(maxerr,err)
            for m in METHODS:
                for label,neg in [('addition_all',lambda r:~r['y']),('addition_normal_action',lambda r:(r['gt']>0)&~r['y']),('addition_background',lambda r:(r['gt']==0)&~r['y'])]:
                    mask=lambda r:((r['gt']<0)&r['y'])|neg(r)
                    vals=evaluate(kernel(rows,m,mask),weights);arr[m][label]=vals
                    yy=np.concatenate([r['y'][mask(r)] for r in rows]);ss=np.concatenate([r[m][mask(r)] for r in rows]);direct=independent_auc(yy,ss)
                    assert abs(direct-vals[0])<1e-12;maxerr=max(maxerr,abs(direct-vals[0]));metric[m][label]=dict(auc=direct,frames=len(yy),positive=int(yy.sum()))
            for m in ['NK','L_HARD']:
                for k in ['whole','known','within']:
                    for draw in [0,1]:
                        err=abs(explicit(rows,weights[draw],m,k)-arr[m][k][draw]);assert err<1e-12;maxerr=max(maxerr,err)
            gt=np.concatenate([r['gt'] for r in rows]);y=np.concatenate([r['y'] for r in rows]);typ=np.concatenate([r['types'] for r in rows]);assert auc(y,np.full(len(y),.5))==.5
            dec={m:decomposition(gt,y,np.concatenate([r[m] for r in rows])) for m in METHODS}
            zero={name:{m:float((np.concatenate([r[m] for r in rows])[mask]==0).mean()) for m in METHODS} for name,mask in dict(addition=(gt<0)&y,normal_action=(gt>0)&~y,background=(gt==0)&~y).items()}
            pairs={a+'__minus__'+b:{k:100*(dec[a]['groups'][k]['contribution']-dec[b]['groups'][k]['contribution']) for k in dec[a]['groups']} for a,b in PAIRS}
            bytype={str(int(t)):{m:auc(y[(typ==0)|(typ==t)],np.concatenate([r[m] for r in rows])[(typ==0)|(typ==t)]) for m in METHODS} for t in np.unique(typ[y])}
            normal={r['video']:{m:dict(mean=float(r[m].mean()),above_half=float((r[m]>=.5).mean())) for m in METHODS} for r in rows if '_normal_' in r['video']}
            output[s]=dict(metrics=metric,contrasts={a+'__minus__'+b:contrast(arr,a,b) for a,b in PAIRS},decomposition=dec,
                pair_differences_pp=pairs,zero_score_rates=zero,error_types=bytype,normal_videos=normal,
                addition_type_counts={str(int(t)):int(((gt<0)&y&(typ==t)).sum()) for t in np.unique(typ[(gt<0)&y])})
            bundles.append(arr)
            for m in METHODS:
                for k,v in arr[m].items():saved[f'{mode}_s{s}_{m}_{k}']=v
            print('ANALYZED',mode,s,flush=True)
        mean=shared_mean(bundles);jm=joint(bundles,20261007)
        analyses[mode]=dict(seeds=output,mean_auc={m:{k:float(v[0]) for k,v in d.items()} for m,d in mean.items()},
            video_contrasts={a+'__minus__'+b:contrast(mean,a,b) for a,b in PAIRS},joint_contrasts={a+'__minus__'+b:contrast(jm,a,b) for a,b in PAIRS},
            mean_pair_differences_pp={a+'__minus__'+b:{k:float(np.mean([o['pair_differences_pp'][a+'__minus__'+b][k] for o in output.values()])) for k in next(iter(output.values()))['pair_differences_pp'][a+'__minus__'+b]} for a,b in PAIRS})
        # Two-task summary is descriptive: coffee selected the candidate.
        ca=np.load(HERE/'actionness_gate_runtime/bootstrap_arrays.npz');common=['NK','LK','E_L','L_HARD']
        cb=[{m:{k:ca[f'{mode}_s{s}_{m}_{k}'] for k in ['whole','known','within']} for m in common} for s in [0,1]]
        cj=joint(cb,20261008)
        combined={m:{k:(jm[m][k]+cj[m][k])/2 for k in cj[m]} for m in common}
        analyses[mode]['two_task_descriptive_joint']={a+'__minus__'+b:contrast(combined,a,b) for a,b in [('L_HARD','NK'),('L_HARD','LK'),('L_HARD','E_L')]}
    co=analyses['median21']['video_contrasts']
    gain=all(co['L_HARD__minus__'+b]['whole']['ci95_pp'][0]>0 for b in ['NK','LK'])
    protect=all(co['L_HARD__minus__'+b][k]['delta_pp']>=0 for b in ['NK','LK'] for k in ['known','within','addition_all'])
    verdict=dict(overall_gain_pass=gain,point_protection_pass=protect,development_marker_pass=gain and protect,
        scope='Frozen candidate transfer to previously used development task; only two fixed supports, no formal noninferiority.')
    np.savez_compressed(OUT/'bootstrap_arrays.npz',weights=weights,**saved)
    io.save(OUT/'results.json',dict(analyses=analyses,mechanisms=mechanism,decision=verdict))
    assert all(io.sha(ROOT/p)==h for p,h in reg['hashes'].items())
    io.save(OUT/'verification.json',dict(complete=True,input_hashes_verified=True,max_independent_auc_bootstrap_error=maxerr,constant_auc=.5,
        seconds=time.time()-start,result_sha=io.sha(OUT/'results.json'),bootstrap_sha=io.sha(OUT/'bootstrap_arrays.npz')))
    print('DECISION',verdict,flush=True)
    for m,d in analyses['median21']['mean_auc'].items():print(m,{k:round(v*100,3) for k,v in d.items()},flush=True)

if __name__=='__main__':main()
