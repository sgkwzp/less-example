"""Standalone metric adapter for the few-shot protocol.

The primary AUC is sklearn's standard frame ROC-AUC.  AEM's historical metric
is retained only as a compatibility calculation and is intentionally exposed
under a different name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, roc_auc_score


def exact_frame_auc(labels, scores):
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    scores = np.asarray(scores).astype(np.float64).reshape(-1)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def aem_legacy_auc(fprs, tprs):
    fprs = np.sort(np.asarray(fprs, dtype=np.float64).reshape(-1))
    tprs = np.sort(np.asarray(tprs, dtype=np.float64).reshape(-1))
    return float(auc(np.concatenate([fprs, [1.0]]), np.concatenate([tprs, [1.0]])))


def aem_threshold_grid():
    values = []
    threshold = 0.0
    while threshold <= 1.0:
        values.append(threshold)
        threshold += 0.025
    return values


def validate_npz(path: Path):
    data = np.load(path)
    labels = data["labels"]
    scores = data["scores"]
    exact = exact_frame_auc(labels, scores)
    result = {
        "file": str(path),
        "frames": int(labels.size),
        "positive_frames": int(np.asarray(labels).sum()),
        "exact_frame_roc_auc": exact,
        "aem_legacy_threshold_count": len(aem_threshold_grid()),
        "aem_legacy_thresholds_last": aem_threshold_grid()[-1],
    }
    return result


def self_test():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(exact_frame_auc(labels, scores) - 1.0) < 1e-12
    grid = aem_threshold_grid()
    assert len(grid) == 40 and abs(grid[-1] - 0.975) < 1e-12
    legacy = aem_legacy_auc([0.0, 0.5, 1.0], [0.0, 0.75, 1.0])
    assert 0.0 <= legacy <= 1.0
    print(json.dumps({"exact_auc": 1.0, "legacy_threshold_count": len(grid), "legacy_auc_fixture": legacy}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--npz", type=Path)
    args = ap.parse_args()
    if args.self_test:
        self_test()
    elif args.npz:
        print(json.dumps(validate_npz(args.npz), indent=2, ensure_ascii=False))
    else:
        ap.error("use --self-test or --npz")


if __name__ == "__main__":
    main()
