#!/usr/bin/env python3
"""
Run the unified algorithms on one system (4 algorithms x 3 scenarios x 2 modes x 30 runs).
Usage: python scripts/run_unified.py <30|118> [n_workers] [ALG,ALG,...|all] [v1|v2]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import json
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'simulation'))
import moeld_algorithms as algs

SCENARIOS = {'Low (30%)': 30, 'High (50%)': 50, 'Ultra-High (70%)': 70}
MODES = {'Deterministic': False, 'Robust': True}
N_RUNS = 30
ORDER = ['MOEA/D', 'HADE-NS', 'NSGA-II', 'MOPSO']  # slowest first


def seed_for(run_id, pen, robust):
    return run_id * 1000 + pen * 10 + (1 if robust else 0)


OUT_NAME = {'v1': 'fronts_{}_unified.json', 'v2': 'fronts_{}_unified_v2.json'}


def run_one(args):
    system, alg, scenario, mode, run_id, policy = args
    pen, robust = SCENARIOS[scenario], MODES[mode]
    seed = seed_for(run_id, pen, robust)
    r = algs.run(system, alg, pen, robust, seed, archive_policy=policy)
    return run_id, {'seed': seed, 'front': r['F'].tolist(), 'cv': r['cv'].tolist(),
                    'time': r['time'], 'nfe': r['nfe'], 'n_gen': r['n_gen']}


def main():
    system = sys.argv[1] if len(sys.argv) > 1 else ''
    if system not in ('30', '118'):
        sys.exit("usage: python scripts/run_unified.py <30|118> [n_workers] [ALG,ALG,...|all] [v1|v2]")
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) - 1)
    chosen = sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3] != 'all' else ORDER
    if any(a not in algs.ALGORITHMS for a in chosen):
        sys.exit(f"algorithms must be from {list(algs.ALGORITHMS)}")
    policy = sys.argv[4] if len(sys.argv) > 4 else 'v2'
    if policy not in OUT_NAME:
        sys.exit(f"archive policy must be one of {list(OUT_NAME)}")
    out_file = os.path.join(ROOT, 'results', OUT_NAME[policy].format(system))
    print(f"archive policy {policy} -> {os.path.relpath(out_file, ROOT)}", flush=True)

    data = {}
    if os.path.exists(out_file):
        with open(out_file) as f:
            data = json.load(f)

    t_start = time.time()
    with Pool(n_workers) as pool:
        for alg in [a for a in ORDER if a in chosen]:
            for scenario in SCENARIOS:
                for mode in MODES:
                    key = f"{scenario}|{mode}|{alg}"
                    if key in data and len(data[key]) == N_RUNS:
                        print(f"[skip] {system} {key}", flush=True)
                        continue
                    t0 = time.time()
                    res = pool.map(run_one, [(system, alg, scenario, mode, r, policy) for r in range(N_RUNS)])
                    data[key] = [x for _, x in sorted(res, key=lambda y: y[0])]
                    tmp = out_file + '.tmp'
                    with open(tmp, 'w') as f:
                        json.dump(data, f)
                    os.replace(tmp, out_file)
                    n_feas = sum(len(r['front']) > 0 for r in data[key])
                    print(f"[done] {system} {key}: {time.time() - t0:.0f}s, runs with a feasible solution "
                          f"{n_feas}/{N_RUNS} (total {(time.time() - t_start) / 60:.1f} min)", flush=True)
    print(f"Saved {out_file}")


if __name__ == '__main__':
    main()
