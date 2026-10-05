#!/usr/bin/env python3
"""
IEEE 30-bus exact Pareto front (epsilon-constraint QP) and IGD+ of every saved run.
Usage: python scripts/exact_front_igd_30.py [n_pts] [fronts file ...]
"""
import json
import os
import sys

import cvxpy as cp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'simulation'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from gap_to_optimum import default_files, dispatch_model
from metrics import ALGORITHMS, igd_plus
from moeld_systems import FEAS_TOL, GAMMA, SYSTEMS

SCENARIOS = {'Low (30%)': 30, 'High (50%)': 50, 'Ultra-High (70%)': 70}
MODES = ['Deterministic', 'Robust']
S = SYSTEMS['30']


def exact_front(pen, robust, n_pts):
    P, base, f, D, re_avail, re_eff = dispatch_model(S, pen, robust)
    cost, emis = f['cost'], f['emission']
    cp.Problem(cp.Minimize(cost), base).solve(solver=cp.CLARABEL)
    e_hi = emis.value
    cp.Problem(cp.Minimize(emis), base).solve(solver=cp.CLARABEL)
    e_lo = emis.value
    pts = []
    for eps in np.linspace(e_lo, e_hi, n_pts):
        cp.Problem(cp.Minimize(cost), base + [emis <= eps + 1e-9]).solve(solver=cp.CLARABEL)
        x = np.clip(P.value, S.P_MIN[:, None], S.P_MAX[:, None])
        o, cv = S.evaluate(x.reshape(1, -1), D, re_avail, robust, GAMMA if robust else 0)
        if cv[0] <= FEAS_TOL:
            pts.append(o[0])
    return np.array(pts), float(re_eff.sum())


def main():
    n_pts = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    files = sys.argv[2:] or default_files('30')
    runs = {}
    for path in files:
        with open(os.path.join(ROOT, 'results', path)) as fh:
            runs.update(json.load(fh))
    out = {'files': files, 'n_front_points': n_pts, 'groups': {}}
    for sc, pen in SCENARIOS.items():
        for mode in MODES:
            key = f"{sc}|{mode}"
            ref, re_energy = exact_front(pen, mode == 'Robust', n_pts)
            ideal, nadir = ref.min(axis=0), ref.max(axis=0)
            span = nadir - ideal
            span[2] = re_energy   # curtailment: fixed on the exact front
            refn = (ref - ideal) / span
            g = {'exact_front': ref.tolist(), 'normalisation_span': span.tolist(), 'algorithms': {}}
            line = []
            for a in ALGORITHMS:
                vals = []
                for r in runs.get(f"{key}|{a}", []):
                    fr = np.array(r['front'])
                    if 'cv' in r:
                        fr = fr[np.array(r['cv']) <= FEAS_TOL]
                    vals.append(igd_plus((fr - ideal) / span, refn) if len(fr) else np.nan)
                g['algorithms'][a] = {'igd_plus_runs': vals, 'igd_plus_median': float(np.nanmedian(vals))}
                line.append(f"{a} {np.nanmedian(vals):.3f}")
            order = sorted(ALGORITHMS, key=lambda a: g['algorithms'][a]['igd_plus_median'])
            g['rank_igd_plus'] = order
            out['groups'][key] = g
            print(f"{key:<30} exact front {len(ref)} pts | IGD+ median: " + ", ".join(line)
                  + f" | rank: {' > '.join(order)}")
    with open(os.path.join(ROOT, 'results', 'igd_plus_30.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print("Saved results/igd_plus_30.json")


if __name__ == '__main__':
    main()
