#!/usr/bin/env python3
"""
Exact optimum of the dispatch model (convex QP) and each algorithm's gap to it.
Usage: python scripts/gap_to_optimum.py [30|118] [fronts file ...]
"""
import json
import os
import sys

import cvxpy as cp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'simulation'))
from moeld_systems import FEAS_TOL, GAMMA, SYSTEMS

SCENARIOS = {'Low (30%)': 30, 'High (50%)': 50, 'Ultra-High (70%)': 70}
MODES = ['Deterministic', 'Robust']
ALGORITHMS = ['HADE-NS', 'NSGA-II', 'MOPSO', 'MOEA/D']


def default_files(system):
    v2 = f'fronts_{system}_unified_v2.json'
    return [v2] if os.path.exists(os.path.join(ROOT, 'results', v2)) else [f'fronts_{system}_unified.json']


def dispatch_model(S, pen, robust):
    """cvxpy variable, constraints and the three objectives of the convex dispatch model."""
    D = S.demand()
    re_avail = S.re_avail(pen)
    re_eff = S.effective_re(re_avail, robust, GAMMA)
    N, T = S.N_GEN, S.HOURS
    (ca, cb, cc), (ea, eb, ec) = S.COST, S.EMIT
    if np.any(cc < 0) or np.any(ec < 0):
        raise ValueError("negative quadratic coefficient: model is not convex")
    P = cp.Variable((N, T))
    Pacc = D - cp.sum(P, axis=0)
    cons = [P >= S.P_MIN[:, None], P <= S.P_MAX[:, None], Pacc >= 0, Pacc <= re_eff,
            cp.sum(S.P_MAX[:, None] - P, axis=0) >= S.RESERVE * D,
            cp.abs(P[:, 1:] - P[:, :-1]) <= S.RAMP[:, None]]
    f = {'cost': T * ca.sum() + cp.sum(cb @ P) + cp.sum(cp.multiply(cc[:, None], cp.square(P))),
         'emission': T * ea.sum() + cp.sum(eb @ P) + cp.sum(cp.multiply(ec[:, None], cp.square(P))),
         'curtailment': cp.sum(re_eff - Pacc)}
    return P, cons, f, D, re_avail, re_eff


def exact_optimum(S, pen, robust):
    P, cons, f, D, re_avail, _ = dispatch_model(S, pen, robust)
    out = {}
    for name, obj in f.items():
        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver=cp.CLARABEL)
        x = np.clip(P.value, S.P_MIN[:, None], S.P_MAX[:, None])
        vals, cv = S.evaluate(x.reshape(1, -1), D, re_avail, robust, GAMMA if robust else 0)
        out[f'min_{name}'] = {'status': prob.status, 'cost': float(vals[0, 0]), 'emission': float(vals[0, 1]),
                              'curtailment': float(vals[0, 2]), 'cv': float(cv[0])}
    return out


def main():
    system = sys.argv[1] if len(sys.argv) > 1 else '118'
    files = sys.argv[2:] or default_files(system)
    S = SYSTEMS[system]
    runs = {}
    for path in files:
        with open(os.path.join(ROOT, 'results', path)) as fh:
            runs.update(json.load(fh))

    res = {'system': system, 'files': files, 'groups': {}}
    for sc, pen in SCENARIOS.items():
        for mode in MODES:
            key = f"{sc}|{mode}"
            opt = exact_optimum(S, pen, mode == 'Robust')
            c_star, e_star, k_star = (opt['min_cost']['cost'], opt['min_emission']['emission'],
                                      opt['min_curtailment']['curtailment'])
            g = {'optimum': opt, 'algorithms': {}}
            print(f"\n=== {key} ===  exact: cost {c_star:,.1f}  emission {e_star:,.3f}  "
                  f"curtailment floor {k_star:,.1f} (curtailment at min-cost point "
                  f"{opt['min_cost']['curtailment']:,.1f})")
            print(f"{'algorithm':<9}{'cost gap med':>13}{'best run':>10}{'curt med':>12}{'curt excess':>13}")
            for a in ALGORITHMS:
                cg, kc = [], []
                for r in runs.get(f"{key}|{a}", []):
                    fr = np.array(r['front'])
                    if 'cv' in r:
                        fr = fr[np.array(r['cv']) <= FEAS_TOL]
                    if len(fr):
                        cg.append((fr[:, 0].min() - c_star) / c_star * 100)
                        kc.append(fr[:, 2].min())
                if not cg:
                    continue
                g['algorithms'][a] = {'cost_gap_pct_runs': cg, 'cost_gap_pct_median': float(np.median(cg)),
                                      'cost_gap_pct_best': float(np.min(cg)),
                                      'min_curtailment_runs': kc, 'min_curtailment_median': float(np.median(kc))}
                print(f"{a:<9}{np.median(cg):12.2f}%{np.min(cg):9.2f}%{np.median(kc):12,.1f}"
                      f"{np.median(kc) - k_star:+13,.1f}")
            res['groups'][key] = g
    out = os.path.join(ROOT, 'results', f'gap_to_optimum_{system}.json')
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1)
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
