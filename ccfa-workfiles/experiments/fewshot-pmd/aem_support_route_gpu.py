"""Frozen K2 AEM scorer for registered support-only routes."""
from pathlib import Path
import time
import numpy as np
import aem_cross_route_gpu as e

def main():
    root=Path('/hy-tmp');here=root/'ccfa-workfiles/experiments/fewshot-pmd';out=here/'aem_support_route_runtime'
    reg=e.read(out/'registered.json');routes=e.read(out/'routes_complete.json');assert routes['complete']
    # Same GPU/model environment as C; verify every inherited registered input.
    prior=e.read(here/'aem_cross_route_runtime/gpu/registered.json')
    assert all(e.sha(p)==h for p,h in prior['hashes'].items())
    for rel,h in reg['hashes'].items():
        # graph DP and preparation-only files need not be installed remotely.
        if Path(rel).name in ['aem_support_route_gpu.py','aem_cross_route_gpu.py','aem-support-route-protocol.md'] or '/replay/C/' in rel:
            assert e.sha(root/rel)==h,rel
    assert all(e.sha(out/p)==h for p,h in routes['hashes'].items())
    assert not (out/'gpu_complete.json').exists()
    torch,clip,tokens=e.initialize(root,out,1);start=time.time();checks=[]
    for j in reg['jobs']:
        m,ds=e.model_data(j,clip)
        for i in range(len(ds)):
            video=ds[i];vid=video['video_id']
            native=dict(np.load(here/'aem_cross_route_runtime/replay/C'/f"s{j['seed']}_KK"/(vid+'.npz')))
            route=dict(np.load(out/'routes'/f"s{j['seed']}"/(vid+'.npz')))
            assert np.array_equal(route['pred_action']==0,native['pred_action']==0)
            route={k:route[k] for k in ['segments','segment_action_pred','pred_action']}
            route.update(labels=native['labels'],gt_action=native['gt_action'])
            with torch.no_grad():
                control=e.frame_scores(native,e.rescore(m,video,native,clip,tokens))
                err=float(np.abs(control['scores']-native['scores']).max());assert err<=1e-6
                scores=e.frame_scores(route,e.rescore(m,video,route,clip,tokens))
            assert np.isfinite(scores['scores']).all()
            dest=out/'scores'/f"s{j['seed']}"/(vid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
            np.savez_compressed(dest,**route,**scores)
            checks.append(dict(seed=j['seed'],video=vid,control_error=err))
        del m;torch.cuda.empty_cache()
    hashes={p.relative_to(out).as_posix():e.sha(p) for p in sorted((out/'scores').glob('*/*.npz'))}
    e.save(out/'gpu_complete.json',dict(complete=True,core_seconds=time.time()-start,checks=checks,hashes=hashes,
        registration_sha=e.sha(out/'registered.json'),routes_manifest_sha=e.sha(out/'routes_complete.json')))
    print('GPU_COMPLETE',len(hashes),time.time()-start,flush=True)

if __name__=='__main__':main()
