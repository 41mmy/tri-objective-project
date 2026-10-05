#!/usr/bin/env python3
"""Exact 3-D hypervolume, IGD+, and the statistics helpers (Mann-Whitney U, Holm, Cliff's delta)."""
from itertools import combinations

import numpy as np
from scipy.stats import mannwhitneyu

ALGORITHMS = ['HADE-NS', 'NSGA-II', 'MOPSO', 'MOEA/D']


def hv2d(pts, ref):
    pts = pts[np.argsort(pts[:, 0])]
    area, best_y = 0.0, ref[1]
    for x, y in pts:
        if y < best_y:
            area += (ref[0] - x) * (best_y - y)
            best_y = y
    return area


def hv3d(front, ref):
    pts = front[np.all(front < ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 2])]
    vol = 0.0
    for i in range(len(pts)):
        z_next = pts[i + 1, 2] if i + 1 < len(pts) else ref[2]
        if z_next > pts[i, 2]:
            vol += hv2d(pts[:i + 1, :2], ref[:2]) * (z_next - pts[i, 2])
    return vol


def igd_plus(approx, ref):
    d = np.maximum(approx[None, :, :] - ref[:, None, :], 0.0)
    return float(np.sqrt((d ** 2).sum(axis=2)).min(axis=1).mean())


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float((np.sum(a[:, None] > b[None, :]) - np.sum(a[:, None] < b[None, :])) / (len(a) * len(b)))


def magnitude(d):
    d = abs(d)
    return 'negligible' if d < 0.147 else 'small' if d < 0.33 else 'medium' if d < 0.474 else 'large'


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank_, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank_) * pvals[idx]))
        adj[idx] = running
    return adj


def compare(values, higher_is_better):
    pairs = list(combinations(ALGORITHMS, 2))
    raw = []
    for a, b in pairs:
        _, p = mannwhitneyu(values[a], values[b], alternative='two-sided')
        raw.append(p)
    adj = holm(np.array(raw))
    out = {}
    for (a, b), p, pa in zip(pairs, raw, adj):
        d = cliffs_delta(values[a], values[b])
        better_a = d > 0 if higher_is_better else d < 0
        out[f"{a} vs {b}"] = {
            'p': float(p), 'p_holm': float(pa), 'sig_holm': bool(pa < 0.05),
            'cliffs_delta': d, 'magnitude': magnitude(d),
            'favors': (a if better_a else b) if d != 0 else 'tie',
        }
    return out


def rank(medians, higher_is_better):
    return sorted(medians, key=lambda k: -medians[k] if higher_is_better else medians[k])
