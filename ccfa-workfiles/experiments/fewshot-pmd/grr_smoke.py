"""Native VideoCLIP reference/graph smoke for EgoPER, not full GRR+SSN.

It is deliberately small and auditable: support labels come from the existing
EgoPER annotation file, no network is trained, and the GTG2Vid generalized DAG
DP and drop-cost definitions are reused for the graph-constrained assignment.
The capped reference bank and in-sample scale below are smoke-only placeholders;
this script does not implement within-step DTW or leave-one-video-out SSN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "code" / "AEM" / "data" / "egoper"
RUNTIME = Path(__file__).resolve().parent / "g0_runtime"
JOBS = RUNTIME.parent / "g1_runtime" / "jobs.jsonl"
FEATURE_TRACK = "videoclip256_10fps"
FEATURE_SUBDIR = "vc_v_features_10fps"
sys.path.insert(0, str(ROOT / "code" / "GTG2Vid"))
from dp.graph_utils import compute_generalized_metadag_costs, generalized_metadag2vid  # noqa: E402
from datasets.loader_graph import GraphLoader  # noqa: E402


def load_job(path, recipe, k, seed):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [r for r in rows if r["method"] == "gtg" and r["recipe"] == recipe
               and str(r["K"]) == str(k) and r["seed"] == seed]
    if len(matches) != 1:
        raise ValueError("Expected one GTG job in the synchronized G1 manifest")
    job = matches[0]
    if job["feature_track"] != FEATURE_TRACK or job["fps"] != 10:
        raise ValueError("Smoke requires the locked native256/10fps job")
    support = job["support_ids"]
    digest = hashlib.sha256(("\n".join(support) + "\n").encode()).hexdigest()[:16]
    if digest != job["support_hash"] or len(support) != k or len(set(support)) != k:
        raise ValueError("Support IDs/hash do not match the G1 manifest")
    if set(support) & set(job["test"]):
        raise ValueError("Support/test overlap")
    return job


def load_features(recipe, video_id):
    path = DATA / recipe / FEATURE_SUBDIR / f"{video_id}.npy"
    x = np.load(path).astype("float32")
    if x.ndim != 2 or x.shape[1] != 256 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError(f"Expected finite nonempty [T,256] native features: {path}")
    return x


def drop_config(job, override=None, runtime=JOBS.parent):
    # Prefer the exact materialized job config when available on this host.
    local = runtime / "configs" / "gtg" / Path(job["config"]).name
    sources = {"coffee": "vc_4omini_post_db0.3_ndb0.json", "oatmeal": "vc_4omini_post_db0.4.json",
               "pinwheels": "vc_4omini_post_db0.3.json", "quesadilla": "vc_4omini_post_db0.2.json",
               "tea": "vc_4omini_post_db0.6.json"}
    path = override or (local if local.is_file() else ROOT / "code/GTG2Vid/configs/EgoPER" / job["recipe"] / sources[job["recipe"]])
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg["dataset_name"] != job["recipe"] or cfg["input_dim"] != 256 or cfg["v_feat_path"] != FEATURE_SUBDIR:
        raise ValueError(f"GTG config recipe/feature mismatch: {path}")
    mapping = json.loads((DATA / "action2idx.json").read_text(encoding="utf-8"))[job["recipe"]]
    classes = max(int(v) for v in mapping.values()) + 1
    # Runner defaults the node constant to gtg_num_classes = 2*num_classes.
    return classes, float(cfg["drop_base"]), float(cfg.get("node_drop_base", 2 * classes)), path


def frame_labels(annotation: dict, n: int, fps: float = 10.0):
    action = np.zeros(n, dtype=np.int64)
    error = np.zeros(n, dtype=np.int64)
    for label, kind, (start, end) in zip(
        annotation["labels"]["action"],
        annotation["labels"]["action_type"],
        annotation["labels"]["time_stamp"],
    ):
        lo = max(0, int(round(start * fps)))
        hi = min(n, int(round(end * fps)) + 1)
        action[lo:hi] = int(label)
        error[lo:hi] = int(kind > 0)
    return action, error


def collect_support(annotation_by_id, support_ids, recipe="coffee", stride=1):
    feats, labels = [], []
    for vid in support_ids:
        x = load_features(recipe, vid)
        y, e = frame_labels(annotation_by_id[vid], len(x))
        assert not e.any(), f"support video is not normal: {vid}"
        feats.append(x[::stride])
        labels.append(y[::stride])
    return feats, labels


def build_reference(feats, labels, max_per_step=96):
    refs = {}
    for x, y in zip(feats, labels):
        for step in np.unique(y):
            rows = x[y == step]
            if len(rows):
                stride = max(1, len(rows) // max_per_step)
                refs.setdefault(int(step), []).append(rows[::stride][:max_per_step])
    out = {}
    for step, chunks in refs.items():
        z = np.concatenate(chunks, axis=0)
        z /= np.linalg.norm(z, axis=1, keepdims=True).clip(min=1e-8)
        out[step] = torch.from_numpy(z)
    return out


def reference_costs(query: np.ndarray, refs: dict[int, torch.Tensor], num_steps: int):
    q = torch.from_numpy(query.astype("float32"))
    q = torch.nn.functional.normalize(q, dim=1)
    sims = torch.full((len(q), num_steps + 1), -1.0)
    for step, bank in refs.items():
        sims[:, step] = (q @ bank.T).max(dim=1).values
    # Background uses the same cosine reference rule as every other step.
    # Unobserved classes retain the common -1 fallback, including background.
    return sims


def graph_costs(sims: torch.Tensor, recipe, drop_base, node_drop_base):
    # GraphLoader is the repository's exact task-graph implementation.
    graph = GraphLoader("EgoPER", recipe, num_classes=sims.shape[1]).graph_info
    # GRR has no learned addition-type classifier.  Use the base task
    # metagraph for the smoke path; it is not GTG's augmented error metagraph.
    gmetadag = graph["metagraph"]
    nodes = list(nx.lexicographical_topological_sort(gmetadag))
    idx2node = {idx: node for idx, node in enumerate(nodes)}
    max_node = max(int(float(v.split(",")[0])) for v in idx2node.values())
    logits = torch.full((max(max_node + 1, sims.shape[1]), sims.shape[0]), -8.0)
    # The normal action ids are the first nodes in the EgoPER task graph.
    logits[: sims.shape[1]] = (sims.T * 8.0)
    sample = {"action_logits": logits}
    # Frame cost = -(normalized entropy + drop_base), not a drop fraction.
    # Node cost = -node_drop_base * (1 - max_t class probability).
    # The upstream -100 sentinel sets frame-drop cost to zero (second pass).
    zx, drop, node_drop = compute_generalized_metadag_costs(
        sample, idx2node, drop_base=drop_base, node_drop_base=node_drop_base)
    return zx, drop, node_drop, gmetadag, idx2node


def graph_assign(sims: torch.Tensor, recipe="coffee", *, drop_base, node_drop_base):
    zx, drop, node_drop, gmetadag, idx2node = graph_costs(sims, recipe, drop_base, node_drop_base)
    _, pred, _ = generalized_metadag2vid(
        zx.cpu().numpy(), drop.cpu().numpy(), node_drop.cpu().numpy(), gmetadag, idx2node
    )
    pred = np.asarray(pred)
    pred[pred >= sims.shape[1]] = 0
    return pred, {"nodes": len(idx2node), "drop_fraction": float(np.mean(pred < 0)),
                  "background_fraction": float(np.mean(pred == 0)),
                  "drop_base": drop_base, "node_drop_base": node_drop_base}


def score_query(query, refs, support_feats, support_labels, assignment):
    q = torch.from_numpy(query.astype("float32"))
    q = torch.nn.functional.normalize(q, dim=1)
    residual = np.ones(len(q), dtype=np.float32)
    for step, bank in refs.items():
        idx = np.where(assignment == step)[0]
        if len(idx):
            residual[idx] = (1.0 - (q[idx] @ bank.T).max(dim=1).values).numpy()
    dropped = assignment < 0
    if dropped.any():
        all_banks = torch.cat(list(refs.values()), dim=0)
        residual[dropped] = (1.0 - (q[dropped] @ all_banks.T).max(dim=1).values).numpy()

    # Smoke-only in-sample scale: bank includes the current support video.
    # This is not cross-video calibration or the proposed SSN estimator.
    centers, scales = {}, {}
    for step, bank in refs.items():
        blocks = []
        for x, y in zip(support_feats, support_labels):
            rows = x[y == step]
            if len(rows):
                rows = torch.from_numpy(rows.astype("float32"))
                rows = torch.nn.functional.normalize(rows, dim=1)
                blocks.append(float((1.0 - (rows @ bank.T).max(dim=1).values).median()))
        if blocks:
            centers[step] = float(np.median(blocks))
            scales[step] = float(max(np.median(np.abs(np.asarray(blocks) - centers[step])), 0.02))
    z = residual.copy()
    for step in np.unique(assignment):
        if step in centers:
            z[assignment == step] = (residual[assignment == step] - centers[step]) / scales[step]
    score = 1.0 / (1.0 + np.exp(-z))
    return score.astype("float32"), residual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="coffee")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=Path, default=JOBS, help="synchronized G1 jobs.jsonl")
    ap.add_argument("--gtg-config", type=Path, help="optional exact GTG config for drop defaults")
    ap.add_argument("--drop-base", type=float, help="frame entropy additive constant; defaults to GTG recipe config")
    ap.add_argument("--node-drop-base", type=float, help="node drop constant; defaults to GTG recipe config")
    ap.add_argument("--check-costs-only", action="store_true", help="check native features and costs without running the expensive Python DP")
    ap.add_argument("--stride", type=int, default=8, help="temporal stride; even stride=1 remains a smoke implementation")
    args = ap.parse_args()
    if args.stride < 1 or args.k < 1:
        ap.error("stride and k must be positive")
    job = load_job(args.jobs, args.recipe, args.k, args.seed)
    classes, drop_base, node_drop_base, config_path = drop_config(job, args.gtg_config, args.jobs.parent)
    drop_base = drop_base if args.drop_base is None else args.drop_base
    node_drop_base = node_drop_base if args.node_drop_base is None else args.node_drop_base
    if not np.isfinite([drop_base, node_drop_base]).all():
        ap.error("drop constants must be finite")
    annotations = json.loads((DATA / "annotation.json").read_text(encoding="utf-8"))[args.recipe]["segments"]
    annotations = {x["video_id"]: x for x in annotations}
    support_feats, support_labels = collect_support(annotations, job["support_ids"], args.recipe, args.stride)
    refs = build_reference(support_feats, support_labels)
    test_ids = [x for x in job["test"] if "_error_" in x]
    query_id = sorted(test_ids)[0]
    query = load_features(args.recipe, query_id)
    _, gt_error_full = frame_labels(annotations[query_id], len(query))
    query, gt_error = query[::args.stride], gt_error_full[::args.stride]
    sims = reference_costs(query, refs, num_steps=classes - 1)
    metadata = {"feature_track": FEATURE_TRACK, "support": job["support_ids"],
                "support_hash": job["support_hash"], "jobs": str(args.jobs),
                "gtg_config": str(config_path), "drop_base": drop_base,
                "node_drop_base": node_drop_base, "stride": args.stride, "paper_use": False}
    if args.check_costs_only:
        zx, drop, node_drop, _, idx2node = graph_costs(sims, args.recipe, drop_base, node_drop_base)
        if not all(torch.isfinite(t).all() for t in (zx, drop, node_drop)):
            raise ValueError("Nonfinite DP costs")
        print(json.dumps({**metadata, "query": query_id, "frames": len(query), "nodes": len(idx2node),
                          "background_reference_frames": len(refs.get(0, [])),
                          "frame_drop_range": [float(drop.min()), float(drop.max())], "dp_executed": False}))
        return
    assignment, graph_info = graph_assign(sims, args.recipe, drop_base=drop_base, node_drop_base=node_drop_base)
    score, residual = score_query(query, refs, support_feats, support_labels, assignment)
    out = RUNTIME / f"grr_{args.recipe}_k{args.k}_seed{args.seed}.npz"
    np.savez_compressed(out, video_id=query_id, scores=score, residual=residual, labels=gt_error,
                        assignment=assignment, dropped=assignment < 0, metadata=json.dumps(metadata))
    print(json.dumps({**metadata, "query": query_id, "frames": len(query), "positive_error_frames": int(gt_error.sum()), "graph": graph_info, "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
