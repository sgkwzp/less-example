"""Compiled exact GTG graph recurrence and exact within-step DTW.

No fastmath, pruning, temporal subsampling, or beam approximation. Parent order
and first-minimum ties match the upstream Python generalized_metadag2vid.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _graph(cost, drop, node_drop, parents, lengths, actions):
    m, k, n = cost.shape
    ns = m + 2
    d = np.zeros((k + 1, n + 1, ns), dtype=np.float64)
    pointer = np.zeros((k + 1, n + 1, ns), dtype=np.int64)
    d[1:, 0, :] = np.inf
    d[0, 1:, :] = np.inf
    d[0, 0, 1:] = np.inf
    d[0, 1:, 1] = np.cumsum(drop)
    d[1:, 0, 2] = np.cumsum(node_drop)
    for x in range(1, n + 1):
        pointer[0, x, 1] = (x - 1) * ns + 1
    for z in range(1, k + 1):
        for x in range(1, n + 1):
            # Best predecessor, preserving parent-list then state tie order.
            pz = parents[z - 1, 0]
            st = np.argmin(d[pz, x - 1])
            best = d[pz, x - 1, st]
            bp = (pz * (n + 1) + x - 1) * ns + st
            for a in range(1, lengths[z - 1]):
                pz = parents[z - 1, a]
                st = np.argmin(d[pz, x - 1])
                if d[pz, x - 1, st] < best:
                    best = d[pz, x - 1, st]
                    bp = (pz * (n + 1) + x - 1) * ns + st
            curst = 0
            for st in range(3, ns):
                if d[z, x - 1, st] < d[z, x - 1, curst]:
                    curst = st
            for a in range(m):
                st = 0 if a == 0 else a + 2
                if d[z, x - 1, curst] < best:
                    d[z, x, st] = d[z, x - 1, curst] + cost[a, z - 1, x - 1]
                    pointer[z, x, st] = (z * (n + 1) + x - 1) * ns + curst
                else:
                    d[z, x, st] = best + cost[a, z - 1, x - 1]
                    pointer[z, x, st] = bp
            st = np.argmin(d[z, x - 1] + drop[x - 1])
            d[z, x, 1] = d[z, x - 1, st] + drop[x - 1]
            pointer[z, x, 1] = (z * (n + 1) + x - 1) * ns + st
            pz = parents[z - 1, 0]
            st = np.argmin(d[pz, x])
            best = d[pz, x, st]
            bp = (pz * (n + 1) + x) * ns + st
            for a in range(1, lengths[z - 1]):
                pz = parents[z - 1, a]
                st = np.argmin(d[pz, x])
                if d[pz, x, st] < best:
                    best = d[pz, x, st]
                    bp = (pz * (n + 1) + x) * ns + st
            d[z, x, 2] = best + node_drop[z - 1]
            pointer[z, x, 2] = bp
    z, x = k, n
    st = np.argmin(d[k, n])
    value = d[k, n, st]
    labels = np.zeros(n, dtype=np.int64)
    types = np.zeros(n, dtype=np.int64)
    meta = np.full(n, -1, dtype=np.int64)
    while x > 0:
        meta[x - 1] = z - 1
        if st == 1:
            labels[x - 1] = -1
            types[x - 1] = 0
        elif st != 2:
            labels[x - 1] = actions[z - 1] if z else -1
            types[x - 1] = st - 2 if st >= 3 else 0
        p = pointer[z, x, st]
        st = p % ns
        p //= ns
        x = p % (n + 1)
        z = p // (n + 1)
    return value, labels, types, meta


def generalized_metadag2vid_fast(cost, drop, node_drop, metadag, idx2node, return_meta_labels=False):
    cost, drop, node_drop = [np.ascontiguousarray(a) for a in (cost, drop, node_drop)]
    m, k, n = cost.shape
    if m < 1 or k < 1 or n < 1 or drop.shape != (n,) or node_drop.shape != (k,):
        raise ValueError('Invalid DP cost dimensions')
    if not all(np.isfinite(a).all() for a in (cost, drop, node_drop)):
        raise ValueError('DP inputs must be finite')
    if len(idx2node) != k or set(idx2node) != set(range(k)):
        raise ValueError('Graph/cost node mismatch')
    reverse = {node: i for i, node in idx2node.items()}
    lists = [[reverse[p] + 1 for p in metadag.pred[idx2node[i]]] or [0] for i in range(k)]
    if any(p > i for i, row in enumerate(lists) for p in row):
        raise ValueError('Nodes must be topologically ordered')
    parents = np.zeros((k, max(map(len, lists))), dtype=np.int64)
    for i, row in enumerate(lists):
        parents[i, :len(row)] = row
    lengths = np.array(list(map(len, lists)), dtype=np.int64)
    actions = np.array([int(float(idx2node[i].split(',')[0])) for i in range(k)])
    value, labels, types, meta = _graph(cost, drop, node_drop, parents, lengths, actions)
    if return_meta_labels:
        return value, labels, types, [idx2node[int(i)] if i >= 0 else -1 for i in meta]
    return value, labels, types


@njit(cache=True)
def dtw_residual(cost):
    """Full closed-end DTW; mean matched distance for each query frame.

    Ties prefer diagonal, then query advance, then reference advance.
    """
    n, m = cost.shape
    previous = np.full(m + 1, np.inf)
    previous[0] = 0.0
    pointer = np.zeros((n, m), dtype=np.uint8)
    for i in range(n):
        current = np.full(m + 1, np.inf)
        for j in range(m):
            best = previous[j]
            direction = 0
            if previous[j + 1] < best:
                best, direction = previous[j + 1], 1
            if current[j] < best:
                best, direction = current[j], 2
            current[j + 1] = best + cost[i, j]
            pointer[i, j] = direction
        previous = current
    total = previous[m]
    residual = np.zeros(n)
    count = np.zeros(n)
    i, j = n - 1, m - 1
    while i >= 0 and j >= 0:
        residual[i] += cost[i, j]
        count[i] += 1
        direction = pointer[i, j]
        if direction == 0:
            i, j = i - 1, j - 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
    return residual / count, total
