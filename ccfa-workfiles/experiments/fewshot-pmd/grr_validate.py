"""Full-resolution, cached five-group K=2 pilot. No training or test tuning."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import networkx as nx
import numpy as np
import torch

from grr_smoke import (DATA, JOBS, FEATURE_SUBDIR, GraphLoader, load_job,
                       load_features, frame_labels, drop_config, compute_generalized_metadag_costs)
from fast_alignment import generalized_metadag2vid_fast, dtw_residual
from unified_eval import exact_frame_auc

HERE = Path(__file__).resolve().parent
METHODS = ['A', 'A2', 'B', 'C', 'D']


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def save_npz(path, **arrays):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as fp:
        np.savez_compressed(fp, **arrays)
    tmp.replace(path)


def segments(labels):
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return [(int(labels[a]), int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


class Engine:
    def __init__(self, recipe, job, config, device):
        self.recipe, self.config, self.device = recipe, config, device
        self.classes, self.drop_base, self.node_base, _ = drop_config(job)
        self.graph = GraphLoader('EgoPER', recipe, self.classes).graph_info['metagraph']
        annotations = json.loads((DATA/'annotation.json').read_text(encoding='utf-8'))[recipe]['segments']
        self.annotations = {v['video_id']: v for v in annotations}
        self.tensors, self.support_labels = {}, {}
        self.elapsed = {'reference_seconds': 0., 'dp_seconds': 0., 'dtw_seconds': 0.}

    def features(self, vid):
        if vid not in self.tensors:
            x = load_features(self.recipe, vid)
            z = torch.from_numpy(x).to(self.device)
            self.tensors[vid] = torch.nn.functional.normalize(z, dim=1)
        return self.tensors[vid]

    def labels(self, vid):
        if vid not in self.support_labels:
            y, error = frame_labels(self.annotations[vid], len(self.features(vid)))
            if error.any():
                raise ValueError(f'Non-normal video in support calibration: {vid}')
            self.support_labels[vid] = y
        return self.support_labels[vid]

    def banks(self, supports):
        out = {}
        for s in range(self.classes):
            chunks = [self.features(v)[torch.as_tensor(np.flatnonzero(self.labels(v)==s), device=self.device)] for v in supports]
            chunks = [c for c in chunks if len(c)]
            if chunks:
                out[s] = torch.cat(chunks)
        return out

    def similarities(self, q, banks):
        t = time.perf_counter()
        out = np.full((len(q), self.classes), -1., dtype=np.float32)
        for s, bank in banks.items():
            k = min(self.config['top_m'], len(bank))
            for start in range(0, len(q), 256):
                # Reference bank is complete. Merge chunk top-k exactly.
                best = torch.full((min(256, len(q)-start), k), -torch.inf, device=self.device)
                for b in range(0, len(bank), 4096):
                    sim = q[start:start+256] @ bank[b:b+4096].T
                    best = torch.topk(torch.cat([best, sim], dim=1), k, dim=1).values
                out[start:start+256, s] = best.mean(1).cpu().numpy()
        self.elapsed['reference_seconds'] += time.perf_counter()-t
        return out

    def align(self, sims, graph, ids):
        t = time.perf_counter()
        sample = {'action_logits': torch.from_numpy(sims.T.copy()) * self.config['similarity_logit_scale']}
        zx, drop, nd = compute_generalized_metadag_costs(sample, ids, self.drop_base, self.node_base)
        first = generalized_metadag2vid_fast(zx.numpy(), drop.numpy(), nd.numpy(), graph, ids)[1]
        zx, drop, nd = compute_generalized_metadag_costs(sample, ids, -100., -200.)
        cost, second, _ = generalized_metadag2vid_fast(zx.numpy(), drop.numpy(), nd.numpy(), graph, ids)
        if (second < 0).any():
            raise ValueError('No-drop assignment left unassigned frames')
        assignment = np.where(first < 0, second, first)
        self.elapsed['dp_seconds'] += time.perf_counter()-t
        return assignment, first < 0, cost / len(sims)

    def within_dtw(self, q, assignment, supports, fallback):
        t = time.perf_counter()
        residual = fallback.copy().astype(np.float64)
        for s, lo, hi in segments(assignment):
            candidates = []
            for vid in supports:
                x = self.features(vid)
                for step, a, b in segments(self.labels(vid)):
                    if step != s:
                        continue
                    # One segment at a time; cost matrix resides on CPU.
                    costs = np.empty((hi-lo, b-a), dtype=np.float32)
                    for start in range(lo, hi, 256):
                        costs[start-lo:min(start+256,hi)-lo] = (1-(q[start:min(start+256,hi)] @ x[a:b].T)).clamp(0, 2).cpu().numpy()
                    r, _ = dtw_residual(costs)
                    candidates.append(r)
            if candidates:
                residual[lo:hi] = np.min(np.stack(candidates), axis=0)
        self.elapsed['dtw_seconds'] += time.perf_counter()-t
        return residual

    def infer(self, vid, supports):
        if vid in supports:
            raise ValueError('Query appears in its reference bank')
        q = self.features(vid)
        sims = self.similarities(q, self.banks(supports))
        ids = dict(enumerate(nx.lexicographical_topological_sort(self.graph)))
        assignment, dropped, _ = self.align(sims, self.graph, ids)
        raw = np.clip(1-sims[np.arange(len(q)), assignment], 0, 2)
        dtw = self.within_dtw(q, assignment, supports, raw)
        choices = []
        for support in supports:
            # The linear control fixes one observed support order, retaining
            # repeated steps. Costs and residual definition match A2.
            order = [s for s, _, _ in segments(self.labels(support)) if s != 0]
            names = [f'{s},{i}' for i, s in enumerate(order)]
            graph = nx.DiGraph()
            graph.add_nodes_from(names)
            graph.add_edges_from(zip(names[:-1], names[1:]))
            la, ld, value = self.align(sims, graph, dict(enumerate(names)))
            choices.append((value, la, ld))
        _, linear, linear_drop = min(choices, key=lambda item: item[0])
        linear_raw = np.clip(1-sims[np.arange(len(q)), linear], 0, 2)
        return {'assignment': assignment, 'dropped': dropped, 'raw': raw, 'dtw': dtw,
                'linear_assignment': linear, 'linear_dropped': linear_drop, 'linear_raw': linear_raw}

    def self_blocks(self, vid):
        x, y = self.features(vid), self.labels(vid)
        result = {}
        for s in np.unique(y):
            idx = np.flatnonzero(y == s)
            bank = x[torch.as_tensor(idx, device=self.device)]
            values = []
            for start in range(0, len(idx), 256):
                query_idx = idx[start:start+256]
                best = torch.full((len(query_idx),), -torch.inf, device=self.device)
                for b in range(0, len(idx), 4096):
                    sim = bank[start:start+256] @ bank[b:b+4096].T
                    excluded = np.abs(query_idx[:,None]-idx[None,b:b+4096]) <= self.config['self_exclusion_frames']
                    sim.masked_fill_(torch.as_tensor(excluded, device=self.device), -torch.inf)
                    best = torch.maximum(best, sim.max(1).values)
                z = (1-best).cpu().numpy()
                values.extend(z[np.isfinite(z)].clip(0,2))
            if values:
                result[int(s)] = np.asarray(values)
        return result


def robust(blocks):
    medians = [float(np.median(b)) for b in blocks if len(b)]
    mads = [float(1.4826*np.median(np.abs(b-np.median(b)))) for b in blocks if len(b)]
    return float(np.median(medians)), float(np.median(mads))


def distributions(records, value_key, label_key):
    result = {}
    for record in records:
        for s in np.unique(record[label_key]):
            result.setdefault(int(s), []).append(np.sort(record[value_key][record[label_key] == s]))
    return result


def quantile_score(values, labels, table):
    out = np.empty(len(values))
    pooled = [b for blocks in table.values() for b in blocks]
    for s in np.unique(labels):
        blocks = table.get(int(s), pooled)
        idx = labels == s
        # Equal block weight avoids pretending correlated frames are videos.
        out[idx] = np.mean([(np.searchsorted(b, values[idx], side='left') + np.searchsorted(b, values[idx], side='right')) / (2*len(b)) for b in blocks], axis=0)
    return out


def ssn_stats(cv, own, config, classes):
    pool_cv = [b for blocks in cv.values() for b in blocks]
    pool_self = [b for blocks in own.values() for b in blocks]
    if not pool_cv or not pool_self:
        raise ValueError('Calibration needs nonempty held-out and self blocks')
    result = {}
    for s in range(classes):
        blocks = cv.get(s, [])
        n = len(blocks)
        mc, sc = robust(blocks or pool_cv)
        ms, ss = robust(own.get(s, pool_self))
        k, rho = config['kappa'], config['rho']
        mu = (n*mc + k*rho*ms)/(n+k)
        sigma = max(config['scale_floor'], np.sqrt((n*sc**2+k*rho**2*ss**2)/(n+k)))
        result[s] = {'mu':mu, 'sigma':sigma, 'n_videos':n,
                     'cv_mu':mc, 'cv_sigma':sc, 'self_mu':ms, 'self_sigma':ss,
                     'cv_fallback':not bool(blocks), 'self_fallback':s not in own}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recipe', default='coffee')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--config', type=Path, default=HERE/'pilot_config.json')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--max-videos', type=int, default=0, help='timing only; partial run has no aggregate AUC')
    ap.add_argument('--max-seconds', type=int, default=1800, help='stop between videos and preserve caches')
    args = ap.parse_args()
    torch.set_num_threads(4)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    job = load_job(JOBS, args.recipe, 2, args.seed)
    run = HERE/'pilot_runtime'/f'{args.recipe}_k2_s{args.seed}'
    run.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(args.config.read_bytes())
    for name in ('grr_validate.py', 'fast_alignment.py', 'grr_smoke.py'):
        digest.update((HERE/name).read_bytes())
    digest.update(json.dumps(job, sort_keys=True).encode())
    # Full content hashes make cache reuse depend on actual features/labels.
    digest.update((DATA/'annotation.json').read_bytes())
    for v in job['support_ids']+job['test']:
        digest.update((DATA/args.recipe/FEATURE_SUBDIR/(v+'.npy')).read_bytes())
    _, db, nb, cfgpath = drop_config(job)
    digest.update(cfgpath.read_bytes())
    identity = digest.hexdigest()
    manifest = run/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['fingerprint'] != identity:
        raise ValueError('Run inputs/code changed; preserve prior run before explicit reset')
    save_json(manifest, {'fingerprint':identity, 'config':config, 'support_ids':job['support_ids'],
                       'support_hash':job['support_hash'], 'test_ids':job['test'],
                       'drop_base':db, 'node_drop_base':nb, 'created_unix':time.time()})
    engine = Engine(args.recipe, job, config, args.device)
    start = time.perf_counter()
    heldout = []
    own = {}
    for v in job['support_ids']:
        other = [u for u in job['support_ids'] if u != v]
        assert v not in other and other
        path = run/f'calibration_{v}.npz'
        if path.exists():
            with np.load(path) as d: record = {k:d[k] for k in d.files}
        else:
            print(f'[calibration] {v} against {other}', flush=True)
            record = engine.infer(v, other)
            save_npz(path, **record)
        heldout.append(record)
        for s, block in engine.self_blocks(v).items():
            own.setdefault(s, []).append(block)
    raw_dist = distributions(heldout, 'raw', 'assignment')
    dtw_dist = distributions(heldout, 'dtw', 'assignment')
    linear_dist = distributions(heldout, 'linear_raw', 'linear_assignment')
    stats = ssn_stats(dtw_dist, own, config, engine.classes)
    save_json(run/'calibration.json', {'stats':stats, 'support_ids':job['support_ids'],
                                     'leave_one_video_out':True, 'self_exclusion_frames':config['self_exclusion_frames']})
    order = job['test']
    # Time the long video first, without looking at its accuracy.
    if args.max_videos:
        order = sorted(order, key=lambda v: len(engine.features(v)), reverse=True)[:args.max_videos]
    for index, v in enumerate(order):
        path = run/f'{v}.npz'
        if path.exists():
            continue
        if time.perf_counter()-start > args.max_seconds:
            print('[budget] stopping between videos; cached work preserved', flush=True)
            break
        t = time.perf_counter()
        rec = engine.infer(v, job['support_ids'])
        # Labels are loaded only after all inference and calibration decisions.
        _, error = frame_labels(engine.annotations[v], len(rec['dtw']))
        rec.update(labels=error, video_id=v, fingerprint=identity)
        rec['A'] = quantile_score(rec['linear_raw'], rec['linear_assignment'], linear_dist)
        rec['A2'] = quantile_score(rec['raw'], rec['assignment'], raw_dist)
        rec['B'] = rec['dtw'].copy()
        rec['C'] = quantile_score(rec['dtw'], rec['assignment'], dtw_dist)
        rec['D'] = np.array([(r-stats[int(s)]['mu'])/stats[int(s)]['sigma'] for r,s in zip(rec['dtw'],rec['assignment'])])
        if not all(np.isfinite(rec[m]).all() and len(rec[m])==len(error) for m in METHODS):
            raise ValueError('Invalid output scores')
        rec['seconds'] = time.perf_counter()-t
        save_npz(path, **rec)
        print(f'[{index+1}/{len(order)}] {v} frames={len(error)} seconds={rec["seconds"]:.2f} drop={rec["dropped"].mean():.3f}', flush=True)
    complete = all((run/f'{v}.npz').exists() for v in job['test'])
    report = {'complete':complete, 'fingerprint':identity, 'scope':'untuned single-recipe K2 pilot',
              'videos_done':sum((run/f'{v}.npz').exists() for v in job['test']),
              'videos_expected':len(job['test']), 'elapsed_this_run':time.perf_counter()-start,
              'stages_this_run':engine.elapsed, 'device':args.device,
              'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2 if args.device=='cuda' else 0}
    if complete:
        records=[]
        for v in job['test']:
            with np.load(run/f'{v}.npz') as d:
                records.append({k:d[k] for k in d.files})
        y=np.concatenate([r['labels'] for r in records])
        report.update(frames=len(y),positive_frames=int(y.sum()),
                      auc={m:exact_frame_auc(y,np.concatenate([r[m] for r in records])) for m in METHODS},
                      drop_fraction=float(np.concatenate([r['dropped'] for r in records]).mean()),
                      inference_seconds=float(sum(r['seconds'] for r in records)),
                      per_video=[{'video':str(r['video_id']), 'frames':len(r['labels']),
                                  'positive_frames':int(r['labels'].sum()),
                                  'auc':{m:(exact_frame_auc(r['labels'],r[m]) if len(np.unique(r['labels']))>1 else None) for m in METHODS}} for r in records])
    save_json(run/'results.json',report)
    print(json.dumps(report if not complete else {k:v for k,v in report.items() if k!='per_video'}),flush=True)


if __name__ == '__main__':
    main()
