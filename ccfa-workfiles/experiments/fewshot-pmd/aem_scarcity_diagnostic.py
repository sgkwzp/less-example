"""CPU-only retrospective diagnostic of immutable historical AEM exports."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / 'aem_scarcity_diagnostic_runtime'
SOURCE = HERE / 'g1_runtime'
PROTOCOL = HERE / 'aem-scarcity-diagnostic-protocol.md'
JOBS = ['aem_coffee_k2_s0', 'aem_coffee_kfull_s0',
        'aem_oatmeal_k2_s0', 'aem_oatmeal_k2_s1']


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def save(p, a):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(a, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def independent_auc(y, s):
    p = int(y.sum()); n = len(y)-p
    return float((rankdata(s)[y].sum()-p*(p+1)/2)/(p*n)) if p*n else None


def kernel(rows, method, mask):
    pos, neg = [], []
    for r in rows:
        m = mask(r)
        pos.append(r[method][m & r['y']])
        neg.append(np.sort(r[method][m & ~r['y']]))
    wins = np.zeros((len(rows), len(rows)))
    for i, p in enumerate(pos):
        for j, n in enumerate(neg):
            if len(p) and len(n):
                wins[i,j] = ((np.searchsorted(n,p,'left')+np.searchsorted(n,p,'right'))*.5).sum()
    return wins, np.array([len(p) for p in pos]), np.array([len(n) for n in neg])


def evaluate(k, weights):
    wins, p, n = k
    den = (weights@p)*(weights@n)
    num = np.einsum('bi,ij,bj->b', weights, wins, weights, optimize=True)
    return np.divide(num, den, out=np.full(len(weights), np.nan), where=den>0)


def category(e, n):
    if e>0 and n>0: return 'same_known' if e==n else 'cross_known'
    if e>0 and n==0: return 'known_error_vs_background'
    if e<=0 and n>0: return 'outside_error_vs_known'
    if e<=0 and n==0: return 'outside_error_vs_background'
    return 'other'


def decomposition(gt, y, s):
    total=int(y.sum())*int((~y).sum()); groups={}; matrix=[]
    for e in np.unique(gt[y]):
        p=s[y & (gt==e)]
        for n in np.unique(gt[~y]):
            q=np.sort(s[~y & (gt==n)]); pairs=len(p)*len(q)
            wins=float(((np.searchsorted(q,p,'left')+np.searchsorted(q,p,'right'))*.5).sum())
            matrix.append(dict(error_gt=int(e),normal_gt=int(n),pairs=pairs,auc=wins/pairs))
            d=groups.setdefault(category(e,n),dict(pairs=0,wins=0.))
            d['pairs']+=pairs; d['wins']+=wins
    for d in groups.values():
        d.update(weight=d['pairs']/total,auc=d['wins']/d['pairs'],contribution=d['wins']/total)
    error=abs(sum(d['contribution'] for d in groups.values())-auc(y,s))
    assert error<1e-12
    return dict(groups=groups,matrix=matrix,reconstruction_error=error)


def annotation_frames(a, n):
    gt=np.zeros(n,np.int64); types=gt.copy()
    for s,t,(start,end) in zip(a['action'],a['action_type'],a['time_stamp']):
        lo=max(0,round(start*10)); hi=min(n,round(end*10)+1)
        gt[lo:hi]=s; types[lo:hi]=t
    return gt,types


def load_job(job, ann, checks):
    d=SOURCE/'aem_audit'/job; rows=[]; diffs=[]; singleton=0; mismatch=0
    recipe=job.split('_')[1]
    amap={v['video_id']:v['labels'] for v in ann[recipe]['segments']}
    for f in sorted(d.glob('*.npz')):
        z=np.load(f); gt,types=annotation_frames(amap[f.stem],len(z['labels']))
        assert np.array_equal(gt,z['gt_action']) and np.array_equal(types>0,z['labels']>0)
        pred=np.full(len(gt),-999); raw=np.zeros(len(gt),np.float32); cover=np.zeros(len(gt),np.int16)
        for (lo,hi),c,sim in zip(z['segments'],z['segment_action_pred'],z['segment_action_sim']):
            assert 0<=lo<=hi<len(gt)
            pred[lo:hi+1]=c;raw[lo:hi+1]=1-float(sim);cover[lo:hi+1]+=1
            singleton+=int(lo==hi)
        assert np.all(cover==1) and np.array_equal(pred,z['pred_action'])
        assert np.allclose(raw,z['raw_all'],atol=1.2e-7,rtol=0)
        assert np.array_equal(z['scores'],z['raw_all']*(pred!=0))
        mismatch+=int(z['native_reconstruction_mismatches'])
        old=np.load(SOURCE/'scores/aem'/job/f.name)
        assert np.array_equal(old['labels'],z['labels'])
        delta=float(np.max(np.abs(old['scores']-z['scores'])))
        assert delta<=1.3e-7 and np.array_equal(old['scores']==0,z['scores']==0)
        diffs.append(delta)
        rows.append(dict(video=f.stem,gt=gt,types=types,y=types>0,pred=pred,
                         native=z['scores'].copy(),ungated=z['raw_all'].copy()))
    assert {r['video'] for r in rows}==set(read(d/'report.json')['job']['test'])
    checks[job]=dict(videos=len(rows),frames=sum(len(r['gt']) for r in rows),
                     max_old_score_difference=max(diffs),singleton_segments=singleton,
                     native_threshold_reconstruction_mismatches=mismatch,
                     expected_singleton_mismatches=singleton*len(read(d/'report.json')['thresholds']))
    assert mismatch==checks[job]['expected_singleton_mismatches']
    return rows


def coverage(rows, mask):
    sizes=[int(mask(r).sum()) for r in rows]
    return dict(frames=sum(sizes),positive=sum(int((mask(r)&r['y']).sum()) for r in rows),
                negative=sum(int((mask(r)&~r['y']).sum()) for r in rows),
                videos=sum(n>0 for n in sizes),error_videos=sum(bool((mask(r)&r['y']).any()) for r in rows))


def routing(rows):
    masks=dict(known_normal=lambda r:(r['gt']>0)&~r['y'],
               known_error=lambda r:(r['gt']>0)&r['y'],
               background_normal=lambda r:(r['gt']==0)&~r['y'],
               unknown=lambda r:r['gt']<0,
               pure_normal_video=lambda r:np.full(len(r['gt']),'_normal_' in r['video']))
    out={}
    for name,mask in masks.items():
        d=coverage(rows,mask);n=d['frames']
        d.update(accuracy=sum(int(((r['pred']==r['gt'])&mask(r)).sum()) for r in rows)/n if n else None,
                 predicted_background_fraction=sum(int(((r['pred']==0)&mask(r)).sum()) for r in rows)/n if n else None)
        out[name]=d
    return out


def metrics_boot(rows, methods, weights, paired=False):
    masks={'whole':lambda r:np.ones(len(r['gt']),bool),'known':lambda r:r['gt']>0}
    if paired:
        masks.update(common=lambda r:r['common'],rest=lambda r:(r['gt']>0)&~r['common'])
    steps=sorted(set(int(s) for r in rows for s in np.unique(r['gt']) if s>0))
    output={}; arrays={}; verification=[]
    for method in methods:
        output[method]={}; arrays[method]={}
        for name,mask in masks.items():
            k=kernel(rows,method,mask); vals=evaluate(k,weights)
            y=np.concatenate([r['y'][mask(r)] for r in rows]);s=np.concatenate([r[method][mask(r)] for r in rows])
            direct=auc(y,s)
            if direct is not None:
                verification.append(abs(direct-independent_auc(y,s)))
                assert abs(direct-vals[0])<1e-12
            output[method][name]=dict(auc=direct,**coverage(rows,mask))
            arrays[method][name]=vals
        numer=np.zeros(len(weights));denom=np.zeros(len(weights));step_out={}
        for step in steps:
            k=kernel(rows,method,lambda r:r['gt']==step);v=evaluate(k,weights)
            size=weights@(k[1]+k[2]);valid=np.isfinite(v)
            numer[valid]+=v[valid]*size[valid];denom[valid]+=size[valid]
            step_out[str(step)]={'auc':float(v[0]) if valid[0] else None,'frames':int(size[0]),'positive':int(k[1].sum())}
        v=np.divide(numer,denom,out=np.full(len(weights),np.nan),where=denom>0)
        output[method]['within']=dict(auc=float(v[0]) if np.isfinite(v[0]) else None,eligible_frames=int(denom[0]),steps=step_out)
        arrays[method]['within']=v
    return output,arrays,max(verification)


def contrast(arrays,a,b):
    out={}
    for metric in arrays[a]:
        v=100*(arrays[a][metric]-arrays[b][metric]);good=np.isfinite(v[1:])
        out[metric]=dict(delta_pp=float(v[0]) if np.isfinite(v[0]) else None,
                         ci95_pp=np.quantile(v[1:][good],[.025,.975]).tolist() if good.any() else None,
                         valid_draws=int(good.sum()))
    return out


def main():
    start=time.time();OUT.mkdir(exist_ok=True)
    inputs=[PROTOCOL,Path(__file__),ROOT/'code/AEM/data/egoper/annotation.json']
    for job in JOBS:
        inputs+=list((SOURCE/'aem_audit'/job).glob('*.npz'))+[SOURCE/'aem_audit'/job/'report.json']
        inputs+=list((SOURCE/'scores/aem'/job).glob('*.npz'))
    for k in ('2','full'):
        inputs+=list((SOURCE/f'scores/aem/aem_coffee_k{k}_s1').glob('*.npz'))
    identity={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(inputs))}
    reg=OUT/'registered.json'
    if reg.exists():
        old=read(reg)['hashes']
        if old!=identity:
            amendment=read(OUT/'engineering_amendment.json')
            changed={k for k in identity if identity[k]!=old.get(k)}
            assert changed=={str(Path(__file__).relative_to(ROOT))}
            assert amendment['old_hash']==old[next(iter(changed))] and amendment['new_hash']==identity[next(iter(changed))]
    else: save(reg,dict(created_unix=time.time(),hashes=identity,bootstrap_draws=2000,seed=20261005))
    ann=read(ROOT/'code/AEM/data/egoper/annotation.json');checks={}
    jobs={j:load_job(j,ann,checks) for j in JOBS}
    result=dict(scope='Historical development caches; not a new fully verified AEM reproduction.',routing={},analyses={},native_seed1={})
    maxerr=0.
    for job,rows in jobs.items():result['routing'][job]=routing(rows)
    for recipe,seed in [('coffee',0),('oatmeal',0),('oatmeal',1)]:
        rows=jobs[f'aem_{recipe}_k2_s{seed}'];full=jobs.get(f'aem_{recipe}_kfull_s{seed}')
        methods=['K_native','K_ungated'];paired=full is not None
        work=[]
        for i,r in enumerate(rows):
            t={k:r[k] for k in ('video','gt','types','y')}
            t.update(K_native=r['native'],K_ungated=r['ungated'])
            if paired:
                f=full[i];assert r['video']==f['video'] and np.array_equal(r['gt'],f['gt'])
                t.update(F_native=f['native'],F_ungated=f['ungated'],
                         K_score_F_gate=r['ungated']*(f['pred']!=0),F_score_K_gate=f['ungated']*(r['pred']!=0),
                         common=(r['gt']>0)&(r['pred']==r['gt'])&(f['pred']==r['gt']))
            work.append(t)
        if paired:methods+=['F_native','F_ungated','K_score_F_gate','F_score_K_gate']
        n=len(work);rng=np.random.default_rng(20261005)
        weights=np.vstack([np.ones(n,dtype=int),rng.multinomial(n,np.full(n,1/n),size=2000)])
        for smooth in (False,True):
            wr=[{**r,**{m:median_filter(r[m],size=21,mode='nearest') if smooth else r[m] for m in methods}} for r in work]
            name=f'{recipe}_s{seed}_'+('median21' if smooth else 'raw')
            print('Analyzing',name,flush=True)
            metric,arr,err=metrics_boot(wr,methods,weights,paired);maxerr=max(maxerr,err)
            comparisons=[('K_ungated','K_native')]
            if paired:comparisons += [('F_native','K_native'),('F_ungated','K_ungated'),('K_score_F_gate','K_native'),('F_native','F_score_K_gate'),('F_ungated','F_native')]
            gt=np.concatenate([r['gt'] for r in wr]);y=np.concatenate([r['y'] for r in wr]);types=np.concatenate([r['types'] for r in wr])
            dec={m:decomposition(gt,y,np.concatenate([r[m] for r in wr])) for m in methods}
            bytype={}
            for t in sorted(set(types[types>0])):
                mask=(types==0)|(types==t)
                bytype[str(t)]={m:auc(y[mask],np.concatenate([r[m] for r in wr])[mask]) for m in methods}
            result['analyses'][name]=dict(metrics=metric,contrasts={a+'__minus__'+b:contrast(arr,a,b) for a,b in comparisons},decomposition=dec,error_types=bytype)
            # Explicit resampled arrays check, independent of pair-matrix implementation.
            ix=np.repeat(np.arange(n),weights[1]); yy=np.concatenate([wr[i]['y'] for i in ix]);ss=np.concatenate([wr[i][methods[0]] for i in ix])
            diff=abs(auc(yy,ss)-arr[methods[0]]['whole'][1]);assert diff<1e-12;maxerr=max(maxerr,diff)
    # Native score-only seed1 corroboration; do not infer missing routing.
    for k in ('2','full'):
        rows=[]
        for f in sorted((SOURCE/f'scores/aem/aem_coffee_k{k}_s1').glob('*.npz')):
            z=np.load(f);a=next(a for a in ann['coffee']['segments'] if a['video_id']==f.stem)
            gt,types=annotation_frames(a['labels'],len(z['labels']))
            assert np.array_equal(types>0,z['labels']>0)
            rows.append(dict(gt=gt,y=types>0,types=types,score=median_filter(z['scores'],size=21,mode='nearest')))
        values,_,err=metrics_boot(rows,['score'],np.ones((1,len(rows)),int))
        result['native_seed1'][k]=values['score'];maxerr=max(maxerr,err)
    checks['max_independent_auc_and_bootstrap_error']=maxerr
    checks['all_hashes_unchanged']=all(sha(ROOT/p)==h for p,h in identity.items())
    assert checks['all_hashes_unchanged']
    checks['seconds']=time.time()-start;checks['gpu_used']=False
    save(OUT/'verification.json',checks);save(OUT/'results.json',result)
    print(json.dumps({'completed':True,'seconds':checks['seconds'],'max_error':maxerr}))


if __name__=='__main__':main()
