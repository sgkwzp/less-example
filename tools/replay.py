"""Recompute the archived Coffee results from full per-video predictions."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'ccfa-workfiles/experiments/fewshot-pmd'
sys.path.insert(0, str(EXP))
from aem_scarcity_diagnostic import metrics_boot, contrast, annotation_frames


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify():
    manifest = read(ROOT / 'provenance/original_files.sha256.json')
    for rel, digest in manifest.items():
        actual = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f'Archived file changed: {rel}')
    print(f'PASS: {len(manifest)} original files verified by SHA256.')


def combine(p, bg, residual):
    # Gate only membership in L_HARD; L_GAP is the archived local repair.
    return {'LK': residual * (~bg), 'L_HARD': (p >= .5) * residual,
            'L_GAP': np.where(bg, p * residual, residual),
            'L_P': p * residual, 'E_L': residual}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/replay.json')
    ap.add_argument('--predictions', type=Path,
                    help='Use freshly inferred p/e_L files; keep frozen NK anchor.')
    args = ap.parse_args()
    verify()
    registered = read(EXP / 'actionness_gate_runtime/registered.json')
    ann = {v['video_id']: v['labels'] for v in read(
        ROOT / 'code/AEM/data/egoper/annotation.json')['coffee']['segments']}
    local = read(EXP / 'actionness_local_repair_runtime/results.json')
    methods = ['NK', 'LK', 'L_HARD', 'L_GAP', 'L_P', 'E_L']
    weights = np.vstack([np.ones(38, int), np.random.default_rng(20261005).multinomial(
        38, np.full(38, 1 / 38), size=2000)])
    result = {}; bundles = []; max_error = 0.
    for job in registered['jobs']:
        seed = job['seed']; rows = []
        assert len(job['test']) == 38 and len(set(job['test'])) == 38
        for vid in job['test']:
            inp = np.load(EXP / f'actionness_gate_runtime/inputs/s{seed}/{vid}.npz')
            score = np.load(EXP / f'actionness_gate_runtime/scores/s{seed}/{vid}.npz')
            native = np.load(EXP / f'aem_cross_route_runtime/replay/C/s{seed}_KK/{vid}.npz')
            if args.predictions:
                fresh = np.load(args.predictions / f's{seed}/{vid}.npz')
                p, residual = fresh['p'], fresh['e_L']
            else:
                p, residual = inp['p'], score['e_L']
            gt, types = annotation_frames(ann[vid], len(p))
            assert np.array_equal(gt, native['gt_action'])
            assert np.array_equal(types > 0, native['labels'] > 0)
            bg = fresh['bg'] if args.predictions else inp['bg']
            scores = dict(NK=native['scores'], **combine(p, bg, residual))
            if not args.predictions:
                archived = np.load(EXP / f'actionness_local_repair_runtime/scores/s{seed}/{vid}.npz')
                assert np.array_equal(scores['L_GAP'], archived['L_GAP_raw'])
            rows.append(dict(video=vid, gt=gt, y=types > 0,
                             **{m: median_filter(scores[m], 21, mode='nearest') for m in methods}))
        metrics, arrays, err = metrics_boot(rows, methods, weights)
        max_error = max(max_error, err)
        if not args.predictions:
            for m in methods:
                for k in ['whole', 'known', 'within']:
                    target = local['analyses']['median21']['seeds'][str(seed)]['metrics'][m][k]['auc']
                    assert abs(metrics[m][k]['auc'] - target) < 1e-12
        result[f's{seed}'] = metrics; bundles.append(arrays)
    mean = {m: {k: np.mean([b[m][k] for b in bundles], axis=0)
                for k in ['whole', 'known', 'within']} for m in methods}
    result['mean_auc_percent'] = {m: {k: float(v[0] * 100) for k, v in data.items()}
                                  for m, data in mean.items()}
    result['comparisons'] = {a + '__minus__' + b: contrast(mean, a, b)
                              for a, b in [('L_HARD', 'NK'), ('L_HARD', 'LK'),
                                           ('L_GAP', 'NK'), ('L_GAP', 'LK')]}
    result['max_independent_auc_error'] = max_error
    result['scope'] = '38 Coffee development videos, two fixed support runs; 2000 video bootstrap draws.'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('method       overall    GT>0      within-step')
    for m, data in result['mean_auc_percent'].items():
        print(f'{m:12s}' + ''.join(f'{data[k]:10.3f}' for k in ['whole', 'known', 'within']))
    print('PASS: archived formulas, annotation grid, seed metrics and independent AUC checks.')
    print(f'Output: {args.out}')


if __name__ == '__main__':
    main()
