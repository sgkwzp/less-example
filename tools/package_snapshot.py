"""Copy the verified experiment snapshot without changing historical files."""
import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    args = ap.parse_args()
    source = args.source.resolve()
    dest = Path(__file__).resolve().parents[1]
    rel = Path('ccfa-workfiles/experiments/fewshot-pmd')
    exp = source / rel
    copied = {}

    def copy(path):
        target = dest / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied[target.relative_to(dest).as_posix()] = sha(path)

    seeds = ['actionness_gate.py', 'actionness_gate_gpu.py',
             'actionness_gate_analyze.py', 'actionness_local_repair.py',
             'aem_support_route.py', 'aem_support_route_gpu.py',
             'aem_support_route_analyze.py', 'hard_gate_transfer.py',
             'hard_gate_transfer_gpu.py', 'hard_gate_transfer_analyze.py']
    pending = list(seeds)
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        path = exp / name
        copy(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                filename = module.split('.')[0] + '.py'
                if (exp / filename).is_file():
                    pending.append(filename)
    for prefix in ['actionness-gate', 'actionness-local-repair',
                   'aem-support-route', 'hard-gate-transfer', 'aem-background-factor']:
        for path in exp.glob(prefix + '*.md'):
            copy(path)
    copy(exp / 'PROJECT_PROGRESS.md')
    for name in ['actionness_gate_runtime', 'actionness_local_repair_runtime',
                 'aem_support_route_runtime', 'hard_gate_transfer_runtime']:
        for path in (exp / name).rglob('*'):
            if path.is_file() and path.suffix in {'.json', '.npz', '.log'}:
                copy(path)
    for path in (exp / 'aem_cross_route_runtime/replay/C').glob('s*_KK/*.npz'):
        copy(path)
    for name in ['gpu/registered.json', 'replay/results.json', 'recovered_assets.json']:
        path = exp / 'aem_cross_route_runtime' / name
        if path.exists():
            copy(path)
    for seed in [0, 1]:
        for recipe in ['coffee']:
            copy(exp / f'g1_runtime/configs/aem/{recipe}_k2_seed{seed}.yaml')
            copy(exp / f'g1_runtime/support/EgoPER/{recipe}/support_k2_seed{seed}.txt')
    for root, folders in [('code/AEM', ['libs', 'configs']),
                          ('code/GTG2Vid', ['dp', 'datasets', 'configs'])]:
        paths = [p for p in (source / root).iterdir() if p.is_file()]
        for folder in folders:
            paths.extend((source / root / folder).rglob('*'))
        for path in paths:
            if (path.is_file() and path.suffix in {'.py', '.cpp', '.h', '.yml', '.yaml', '.md', '.json'}
                    and '__pycache__' not in path.parts):
                copy(path)
    data = source / 'code/AEM/data/egoper'
    for name in ['annotation.json', 'action2idx.json']:
        copy(data / name)
    for recipe in ['coffee', 'oatmeal']:
        for name in ['training.txt', 'test.txt']:
            copy(data / recipe / name)

    registered = json.loads((exp / 'actionness_gate_runtime/registered.json').read_text())
    gpu_reg = json.loads((exp / 'aem_cross_route_runtime/gpu/registered.json').read_text())
    external = []
    paths = {}
    for job in registered['jobs']:
        checkpoint = job['checkpoint'].replace('/hy-tmp/', '')
        paths[checkpoint] = ('checkpoint', gpu_reg['hashes'][job['checkpoint']])
        for video in job['support_ids'] + job['test']:
            feature = f'code/AEM/data/egoper/coffee/vc_v_features_10fps/{video}.npy'
            paths[feature] = ('native256_feature', registered['hashes'][feature])
    for path, (kind, expected) in sorted(paths.items()):
        local = source / path
        actual = sha(local)
        if actual != expected:
            raise ValueError(f'Historical SHA mismatch: {path}')
        external.append(dict(path=path, kind=kind, bytes=local.stat().st_size,
                             sha256=actual, original_server_path='/hy-tmp/' + path))
    for path, digest in gpu_reg['hashes'].items():
        if path.endswith('open_clip_model.safetensors'):
            external.append(dict(path='external/open_clip_model.safetensors',
                                 kind='EVA02_CLIP', bytes=None, sha256=digest,
                                 original_server_path=path,
                                 download='https://huggingface.co/timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k'))
    provenance = dest / 'provenance'
    provenance.mkdir(exist_ok=True)
    (provenance / 'external_assets.json').write_text(json.dumps(external, indent=2) + '\n')
    (provenance / 'original_files.sha256.json').write_text(json.dumps(copied, indent=2) + '\n')
    print(f'Copied {len(copied)} immutable files, {len(seen)} experiment modules.')
    print(f'External known-size assets: {sum(a["bytes"] or 0 for a in external) / 1e9:.3f} GB')


if __name__ == '__main__':
    main()

