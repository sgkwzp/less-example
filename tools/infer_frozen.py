"""Re-score frozen Coffee routes and reproduce the support actionness fit.

Query annotations are never read here. Labels used to construct the saved
support probe come exclusively from the two normal support videos.
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'ccfa-workfiles/experiments/fewshot-pmd'
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT / 'code/AEM'))
from actionness_gate import fit
from aem_cross_route_gpu import rescore, sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset-root', type=Path, required=True,
                    help='Directory containing external_assets.json relative paths.')
    ap.add_argument('--clip-weights', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/fresh')
    ap.add_argument('--routes', type=Path, help='Optional regenerated align_support.py output.')
    args = ap.parse_args()
    args.asset_root = args.asset_root.resolve()
    if args.out.exists():
        raise ValueError('Use a new output directory; do not overwrite previous predictions.')
    for item in read(ROOT / 'provenance/external_assets.json'):
        path = args.clip_weights if item['kind'] == 'EVA02_CLIP' else args.asset_root / item['path']
        if not path.is_file() or sha(path) != item['sha256']:
            raise ValueError(f'Missing or changed external asset: {path}')
    import torch
    import open_clip
    from safetensors.torch import load_file
    from torch.utils.cpp_extension import load
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError('Fresh AEM inference requires a CUDA GPU (tested on 4090).')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['MAX_JOBS'] = '2'
    args.out.mkdir(parents=True)
    build = args.out / 'nms_build'; build.mkdir()
    nms = load(name='nms_1d_cpu', sources=[str(ROOT / 'code/AEM/libs/utils/csrc/nms_cpu.cpp')],
               extra_cflags=['-fopenmp'], extra_ldflags=['-fopenmp'],
               build_directory=str(build), verbose=False)
    sys.modules['libs.utils.nms_1d_cpu'] = nms
    import open_clip.model as clip_impl
    config = open_clip.get_model_config('EVA02-L-14-336')
    custom = config.pop('custom_text', False)
    clip = (clip_impl.CustomTextCLIP if custom else clip_impl.CLIP)(**config)
    clip.load_state_dict(load_file(str(args.clip_weights)), strict=True)
    clip = clip.cuda().eval()
    clip_tuple = (clip, open_clip.get_tokenizer('EVA02-L-14-336'), None)
    from libs.datasets.clip_encode import action_tokenize_egoper
    from libs.core import load_config
    from libs.modeling import make_meta_arch
    from libs.utils import fix_random_seed
    fix_random_seed(0, include_cuda=True)
    reg = read(EXP / 'actionness_gate_runtime/registered.json')
    # Only step names are needed for the text encoder; omit all video labels.
    action2idx = read(ROOT / 'code/AEM/data/egoper/annotation.json')['coffee']['action2idx']
    tokens = action_tokenize_egoper(clip_tuple, 'coffee', {'action2idx': action2idx}, device='cuda:0')
    from grr_smoke import frame_labels
    for job in reg['jobs']:
        seed = job['seed']; xs = []; ys = []
        feature_dir = args.asset_root / 'code/AEM/data/egoper/coffee/vc_v_features_10fps'
        for vid in job['support_ids']:
            x = torch.nn.functional.normalize(torch.from_numpy(
                np.load(feature_dir / (vid + '.npy')).astype(np.float32)), dim=1).numpy()
            y, error = frame_labels(reg['support_annotations'][vid], len(x))
            assert not error.any()
            xs.append(x.astype(np.float64)); ys.append((y > 0).astype(int))
        theta, info = fit(xs, ys)
        old_theta = np.load(EXP / f'actionness_gate_runtime/fits/s{seed}.npz')['theta']
        print(f's{seed} support fit max theta difference: {np.max(np.abs(theta - old_theta)):.3g}', flush=True)
        cfg = copy.deepcopy(load_config(str(EXP / f'g1_runtime/configs/aem/coffee_k2_seed{seed}.yaml')))
        cfg['model']['backbone_type'] = 'convTransformer'
        model = make_meta_arch(cfg['model_name'], **cfg['model']).cuda().eval()
        checkpoint = args.asset_root / job['checkpoint'].replace('/hy-tmp/', '')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        assert state['epoch'] == 135
        model.load_state_dict({k.removeprefix('module.'): v for k, v in state['state_dict'].items()}, strict=True)
        del state
        dest = args.out / f's{seed}'; dest.mkdir()
        for vid in job['test']:
            x = np.load(feature_dir / (vid + '.npy')).astype(np.float32)
            q = torch.nn.functional.normalize(torch.from_numpy(x), dim=1).numpy().astype(np.float64)
            p = expit(q @ theta[:-1] + theta[-1])
            inp = np.load(EXP / f'actionness_gate_runtime/inputs/s{seed}/{vid}.npz')
            route = np.load(EXP / f'aem_support_route_runtime/scores/s{seed}/{vid}.npz')
            if args.routes:
                inp = route = np.load(args.routes / f's{seed}/{vid}.npz')
            video = dict(video_id=vid, feats=torch.from_numpy(x).T, fps=10.,
                         duration=len(x) / 10., feat_stride=1, feat_num_frames=1)
            with torch.no_grad():
                sim = rescore(model, video, {k: route[k] for k in ['segments', 'segment_action_pred']}, clip_tuple, tokens)
                residual = np.empty(len(x), np.float32)
                for (a, b), value in zip(route['segments'], sim):
                    residual[a:b + 1] = 1 - float(value)
                if len(inp['l_gap_segments']):
                    gap = rescore(model, video, dict(segments=inp['l_gap_segments'],
                                  segment_action_pred=inp['l_gap_classes']), clip_tuple, tokens)
                    for (a, b), value in zip(inp['l_gap_segments'], gap):
                        residual[a:b + 1] = 1 - float(value)
            old = np.load(EXP / f'actionness_gate_runtime/scores/s{seed}/{vid}.npz')['e_L']
            err = float(np.max(np.abs(residual - old)))
            np.savez_compressed(dest / (vid + '.npz'), p=p, e_L=residual, bg=inp['bg'], residual_max_difference=err)
            print(f's{seed} {vid}: residual max difference {err:.3g}', flush=True)
        del model; torch.cuda.empty_cache()
    (args.out / 'complete.json').write_text(json.dumps(dict(complete=True,
        torch=torch.__version__, scope='Frozen route inference; no query GT read.'), indent=2))


if __name__ == '__main__':
    main()
