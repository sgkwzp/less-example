"""Locked paired-video evaluation; reads query GT only after inference."""
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import median_filter
from aem_scarcity_diagnostic import (auc,independent_auc,metrics_boot,contrast,decomposition,
                                    annotation_frames,kernel,evaluate)
import aem_cross_route_gpu as io

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'actionness_gate_runtime'
METHODS=['NK','LK','N_P','L_P','E_N','E_L','P','L_HARD','L_AEM_ACT','L_AEM_BG','L_HIST']
COMPARISONS=[('L_P','LK'),('L_P','NK'),('N_P','NK'),('L_P','N_P')]+[('L_P',m) for m in ['E_L','P','L_HARD','L_AEM_ACT','L_AEM_BG','L_HIST']]

def load_rows(j,ann):
    rows=[]
    for vid in j['test']:
        s=j['seed'];inp=np.load(OUT/'inputs'/f's{s}'/(vid+'.npz'));z=np.load(OUT/'scores'/f's{s}'/(vid+'.npz'))
        n=np.load(HERE/'aem_cross_route_runtime/replay/C'/f's{s}_KK'/(vid+'.npz'))
        l=np.load(HERE/'aem_support_route_runtime/scores'/f's{s}'/(vid+'.npz'))
        gt,typ=annotation_frames(ann[vid],len(inp['p']));y=typ>0
        assert np.array_equal(gt,n['gt_action']) and np.array_equal(y,n['labels']>0)
        assert np.array_equal(gt,l['gt_action']) and np.array_equal(y,l['labels']>0)
        p=inp['p'];en=z['e_N'];el=z['e_L'];bg=inp['bg']
        assert np.array_equal(en*(~bg),n['scores']) and np.array_equal(el*(~bg),l['scores'])
        for key,pred,anchor,ename in [('n',inp['n_full_pred'],n,'e_N'),('l',inp['l_full_pred'],l,'e_L')]:
            cover=np.zeros(len(bg),int);reconstruct=anchor['scores'].copy()
            for (a,b),c,sim in zip(inp[key+'_gap_segments'],inp[key+'_gap_classes'],z[key+'_gap_sim']):
                assert bg[a:b+1].all() and (pred[a:b+1]==c).all()
                cover[a:b+1]+=1;reconstruct[a:b+1]=1-float(sim)
            assert np.array_equal(cover,bg.astype(int)) and np.array_equal(reconstruct,z[ename])
        scores=dict(NK=n['scores'],LK=l['scores'],N_P=p*en,L_P=p*el,E_N=en,E_L=el,P=p,
            L_HARD=(p>=.5)*el,L_AEM_ACT=z['aem_act']*el,L_AEM_BG=z['aem_nonbg']*el,L_HIST=(~inp['hist_bg'])*el)
        assert all(np.isfinite(v).all() for v in scores.values())
        rows.append(dict(video=vid,gt=gt,types=typ,y=y,bg=bg,**scores))
    return rows

def summary(y,s):
    return dict(frames=len(s),mean=float(s.mean()) if len(s) else None,
        quantiles=np.quantile(s,[.1,.5,.9]).tolist() if len(s) else [],
        above_half=float((s>=.5).mean()) if len(s) else None,auc=auc(y,s))

def explicit(rows,weights,method,metric):
    ix=np.repeat(np.arange(len(rows)),weights)
    y=np.concatenate([rows[i]['y'] for i in ix]);gt=np.concatenate([rows[i]['gt'] for i in ix]);s=np.concatenate([rows[i][method] for i in ix])
    if metric!='within':
        keep=np.ones(len(y),bool) if metric=='whole' else gt>0
        return independent_auc(y[keep],s[keep])
    vals=[];sizes=[]
    for step in np.unique(gt[gt>0]):
        keep=gt==step;a=independent_auc(y[keep],s[keep])
        if a is not None:vals.append(a);sizes.append(keep.sum())
    return float(np.average(vals,weights=sizes))

def main():
    start=time.time();reg=io.read(OUT/'registered.json');cpu=io.read(OUT/'inputs_complete.json');gpu=io.read(OUT/'gpu_complete.json')
    assert cpu['complete'] and gpu['complete'] and gpu['registration_sha']==io.sha(OUT/'registered.json')
    assert gpu['inputs_manifest_sha']==io.sha(OUT/'inputs_complete.json')
    assert all(io.sha(ROOT/p)==h for p,h in reg['hashes'].items())
    assert all(io.sha(OUT/p)==h for d in [cpu,gpu] for p,h in d['hashes'].items())
    ann={a['video_id']:a['labels'] for a in io.read(ROOT/'code/AEM/data/egoper/annotation.json')['coffee']['segments']}
    assert len(reg['jobs'])==2 and all(j['test']==reg['jobs'][0]['test'] for j in reg['jobs'])
    n=len(reg['jobs'][0]['test']);assert n==38
    weights=np.vstack([np.ones(n,int),np.random.default_rng(20261005).multinomial(n,np.full(n,1/n),size=2000)])
    allrows={str(j['seed']):load_rows(j,ann) for j in reg['jobs']};results={};saved={};maxerr=0.
    mechanisms={};mecharr=[]
    for seed,rows in allrows.items():
        gt=np.concatenate([r['gt'] for r in rows]);y=np.concatenate([r['y'] for r in rows]);bg=np.concatenate([r['bg'] for r in rows]);el=np.concatenate([r['E_L'] for r in rows]);p=np.concatenate([r['P'] for r in rows])
        q2=dict(auc_el_predicting_rejection_within_errors=auc(bg[y],el[y]),
            rejected_error=summary(y[y&bg],el[y&bg]),retained_error=summary(y[y&~bg],el[y&~bg]),
            by_video={r['video']:auc(r['bg'][r['y']],r['E_L'][r['y']]) for r in rows},
            by_step={str(int(s)):auc(bg[y&(gt==s)],el[y&(gt==s)]) for s in np.unique(gt[y])})
        mechrows=[]
        for r in rows:
            mask=((r['gt']>0)&r['y'])|((r['gt']==0)&~r['y'])
            mechrows.append(dict(gt=np.ones(mask.sum()),y=r['y'][mask],P=r['P'][mask],B=(~r['bg'][mask]).astype(float)))
        ma={m:{'whole':evaluate(kernel(mechrows,m,lambda r:np.ones(len(r['y']),bool)),weights)} for m in ['P','B']}
        mecharr.append(ma)
        mechanisms[seed]=dict(Q2=q2,action_vs_background={m:float(ma[m]['whole'][0]) for m in ma},
            action_vs_background_difference=contrast(ma,'P','B'),
            p_groups={name:summary(y[mask],p[mask]) for name,mask in {
                'background_normal':(gt==0)&~y,'action_normal':(gt>0)&~y,'action_error':(gt>0)&y}.items()})
    mm={m:{'whole':np.mean([a[m]['whole'] for a in mecharr],axis=0)} for m in ['P','B']}
    mechanisms['mean_action_vs_background_difference']=contrast(mm,'P','B')
    for mode in ['raw','median21']:
        bundles=[];output={}
        for seed,raw in allrows.items():
            rows=[{**r,**{m:median_filter(r[m],21,mode='nearest') if mode=='median21' else r[m] for m in METHODS}} for r in raw]
            metrics,arr,err=metrics_boot(rows,METHODS,weights);maxerr=max(maxerr,err);bundles.append(arr)
            for m in ['NK','L_P']:
                for metric in ['whole','known','within']:
                    for draw in [0,1]:
                        err=abs(explicit(rows,weights[draw],m,metric)-arr[m][metric][draw]);assert err<1e-12;maxerr=max(maxerr,err)
            gt=np.concatenate([r['gt'] for r in rows]);y=np.concatenate([r['y'] for r in rows]);typ=np.concatenate([r['types'] for r in rows]);bg=np.concatenate([r['bg'] for r in rows])
            assert auc(y,np.full(len(y),.5))==.5
            dec={m:decomposition(gt,y,np.concatenate([r[m] for r in rows])) for m in METHODS}
            pairs={a+'__minus__'+b:{k:100*(dec[a]['groups'][k]['contribution']-dec[b]['groups'][k]['contribution']) for k in dec[a]['groups']} for a,b in COMPARISONS}
            subset={}
            for name,mask in {'rejected':bg,'retained':~bg}.items():
                subset[name]={m:summary(y[mask],np.concatenate([r[m] for r in rows])[mask]) for m in METHODS}
            normal={r['video']:{m:summary(r['y'],r[m]) for m in METHODS} for r in rows if '_normal_' in r['video']}
            bytype={str(int(t)):{m:auc(y[(typ==0)|(typ==t)],np.concatenate([r[m] for r in rows])[(typ==0)|(typ==t)]) for m in METHODS} for t in np.unique(typ[typ>0])}
            output[seed]=dict(metrics=metrics,contrasts={a+'__minus__'+b:contrast(arr,a,b) for a,b in COMPARISONS},
                decomposition=dec,pair_contribution_differences_pp=pairs,subsets=subset,normal_videos=normal,error_types=bytype)
            for m in METHODS:
                for metric in arr[m]:saved[f'{mode}_s{seed}_{m}_{metric}']=arr[m][metric]
            print('ANALYZED',mode,seed,flush=True)
        mean={m:{k:np.mean([a[m][k] for a in bundles],axis=0) for k in bundles[0][m]} for m in METHODS}
        results[mode]=dict(seeds=output,mean_auc={m:{k:float(v[0]) for k,v in mm.items()} for m,mm in mean.items()},
            mean_contrasts={a+'__minus__'+b:contrast(mean,a,b) for a,b in COMPARISONS},
            mean_pair_differences_pp={a+'__minus__'+b:{k:float(np.mean([o['pair_contribution_differences_pp'][a+'__minus__'+b][k] for o in output.values()])) for k in next(iter(output.values()))['pair_contribution_differences_pp'][a+'__minus__'+b]} for a,b in COMPARISONS})
    main=results['median21'];co=main['mean_contrasts']
    overall=all(co['L_P__minus__'+b]['whole']['ci95_pp'][0]>0 for b in ['NK','LK'])
    protect=all(co['L_P__minus__'+b][k]['delta_pp']>=0 for b in ['NK','LK'] for k in ['known','within'])
    verdict=dict(overall_gain_pass=overall,development_point_protection_pass=protect,extension_pass=overall and protect,
        scope='Coffee development only; two fixed support/model seeds; not a confirmation or noninferiority test.')
    np.savez_compressed(OUT/'bootstrap_arrays.npz',weights=weights,**saved)
    io.save(OUT/'results.json',dict(analyses=results,mechanisms=mechanisms,decision=verdict))
    io.save(OUT/'verification.json',dict(complete=True,all_input_hashes_verified=True,constant_auc=.5,
        independent_auc_and_explicit_bootstrap_max_error=maxerr,seconds=time.time()-start,
        result_sha=io.sha(OUT/'results.json'),bootstrap_sha=io.sha(OUT/'bootstrap_arrays.npz'),
        original_segment_control_max_error=max(max(c['control_max_difference'].values()) for c in gpu['checks'])))
    print('DECISION',verdict,flush=True)
    for m,d in main['mean_auc'].items():print(m,{k:round(v*100,4) for k,v in d.items()},flush=True)

if __name__=='__main__':main()
