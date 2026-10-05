"""Regenerate demonstration alignment using only normal support annotations."""
import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'ccfa-workfiles/experiments/fewshot-pmd'
sys.path.insert(0, str(EXP))
from grr_validate import Engine, segments
from actionness_gate import gap_segments


class SupportEngine(Engine):
    def __init__(self, job, annotations, feature_dir):
        self.recipe = 'coffee'
        self.config = dict(top_m=5, similarity_logit_scale=8.)
        self.device = 'cpu'; self.classes = 16
        self.drop_base = .3; self.node_base = 0.
        self.annotations = {v: annotations[v] for v in job['support_ids']}
        self.tensors = {}; self.support_labels = {}
        self.elapsed = dict(reference_seconds=0., dp_seconds=0., dtw_seconds=0.)
        self.feature_dir = feature_dir

    def features(self, vid):
        if vid not in self.tensors:
            x = np.load(self.feature_dir / (vid + '.npy')).astype(np.float32)
            assert x.ndim == 2 and x.shape[1] == 256 and np.isfinite(x).all()
            self.tensors[vid] = torch.nn.functional.normalize(torch.from_numpy(x), dim=1)
        return self.tensors[vid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/routes')
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError('Use a new output directory.')
    torch.set_num_threads(1)
    reg = json.loads((EXP / 'actionness_gate_runtime/registered.json').read_text())
    feature_dir = args.asset_root / 'code/AEM/data/egoper/coffee/vc_v_features_10fps'
    diffs = []
    for job in reg['jobs']:
        seed = job['seed']; engine = SupportEngine(job, reg['support_annotations'], feature_dir)
        banks = engine.banks(job['support_ids']); templates = []
        for vid in job['support_ids']:
            names = [f'{c},{i}' for i, (c, a, b) in enumerate(segments(engine.labels(vid))) if c > 0]
            graph = nx.DiGraph(); graph.add_nodes_from(names)
            graph.add_edges_from(zip(names[:-1], names[1:]))
            templates.append((vid, graph, dict(enumerate(names))))
        dest = args.out / f's{seed}'; dest.mkdir(parents=True)
        for vid in job['test']:
            sim = engine.similarities(engine.features(vid), banks)
            choices = [(engine.align(sim, graph, ids), support) for support, graph, ids in templates]
            (linear, dropped, cost), selected = min(choices, key=lambda x: x[0][2])
            old = np.load(EXP / f'actionness_gate_runtime/inputs/s{seed}/{vid}.npz')
            bg = old['bg']; pred = np.where(bg, 0, linear).astype(np.int64)
            runs = segments(pred)
            seg = np.array([[a, b - 1] for c, a, b in runs], np.int64)
            classes = np.array([c for c, a, b in runs], np.int64)
            gap, gap_class = gap_segments(linear, bg)
            differences = int(np.sum(linear != old['l_full_pred']))
            diffs.append(dict(seed=seed, video=vid, changed_route_frames=differences))
            np.savez_compressed(dest / (vid + '.npz'), segments=seg, segment_action_pred=classes,
                pred_action=pred, l_gap_segments=gap, l_gap_classes=gap_class,
                l_full_pred=linear, bg=bg, selected_support=np.array(selected), alignment_cost=np.array(cost))
            del engine.tensors[vid]
        print(f's{seed}: completed {len(job["test"])} full-video alignments.', flush=True)
    (args.out / 'verification.json').write_text(json.dumps(dict(records=diffs,
        changed_route_frames=sum(r['changed_route_frames'] for r in diffs)), indent=2))
    print('Changed route frames relative to archive:', sum(r['changed_route_frames'] for r in diffs))


if __name__ == '__main__':
    main()
