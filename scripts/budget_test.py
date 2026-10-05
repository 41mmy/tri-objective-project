#!/usr/bin/env python3
"""
Gap to the exact optimum when the unified algorithms get 1x, 4x and 16x the evaluations.
Usage: python scripts/budget_test.py [n_workers]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import json
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'simulation'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import moeld_algorithms as A
from moeld_systems import SYSTEMS, System
from run_unified import SCENARIOS, MODES, seed_for

OUT = os.path.join(ROOT, 'results', 'budget_test.json')
ALGORITHMS = ['HADE-NS', 'NSGA-II', 'MOPSO', 'MOEA/D']
RUNS = range(5)
CASES = [('30', 'Low (30%)', 'Deterministic', (1, 4, 16)),
         ('30', 'Ultra-High (70%)', 'Deterministic', (1, 4, 16)),
         ('118', 'Low (30%)', 'Deterministic', (1, 4))]
COST_WEIGHT = {'30': 1, '118': 4, 'MOEA/D': 3}

for _sys in {c[0] for c in CASES}:
    for _k in sorted({k for c in CASES if c[0] == _sys for k in c[3]}):
        base = SYSTEMS[_sys]
        SYSTEMS[f'{_sys}x{_k}'] = System(base.name, base._mod, base.pop_size, base.n_gen * _k)


def job_key(system, k, scenario, mode, alg, run_id):
    return f"{system}|x{k}|{scenario}|{mode}|{alg}|{run_id}"


def run_job(args):
    system, k, scenario, mode, alg, run_id = args
    seed = seed_for(run_id, SCENARIOS[scenario], MODES[mode])
    # archive policy v1 = the rule of fronts_<system>_unified.json, so x1 reproduces those runs
    r = A.run(f'{system}x{k}', alg, SCENARIOS[scenario], MODES[mode], seed, archive_policy='v1')
    return job_key(*args), {'seed': seed, 'front': r['F'].tolist(), 'cv': r['cv'].tolist(),
                            'nfe': r['nfe'], 'n_gen': r['n_gen'], 'time': r['time']}


def igd_plus(approx, ref):
    d = np.maximum(approx[None, :, :] - ref[:, None, :], 0.0)
    return float(np.sqrt((d ** 2).sum(axis=2)).min(axis=1).mean())


def load(name):
    with open(os.path.join(ROOT, 'results', name)) as f:
        return json.load(f)


def analyse(data):
    exact30 = load('igd_plus_30.json')['groups']
    print(f"\n{'case':<30}{'algorithm':<9}{'budget':>7}{'evaluations':>13}{'cost gap':>10}{'emis gap':>10}"
          f"{'curt+':>8}{'IGD+':>8}   (medians over {len(RUNS)} seeds)")
    summary = {}
    for system, scenario, mode, factors in CASES:
        group = f"{scenario}|{mode}"
        opt = load(f'gap_to_optimum_{system}.json')['groups'][group]['optimum']
        c_star, e_star = opt['min_cost']['cost'], opt['min_emission']['emission']
        k_star = opt['min_curtailment']['curtailment']
        saved = load(f'fronts_{system}_unified.json')
        if system == '30':
            ref = np.array(exact30[group]['exact_front'])
            ideal, span = ref.min(axis=0), np.array(exact30[group]['normalisation_span'])
            ref_n = (ref - ideal) / span
        for alg in ALGORITHMS:
            for k in factors:
                rows = [data[job_key(system, k, scenario, mode, alg, r)] for r in RUNS]
                F = [np.array(x['front']).reshape(-1, 3) for x in rows]
                cg = [100 * (f[:, 0].min() / c_star - 1) for f in F]
                eg = [100 * (f[:, 1].min() / e_star - 1) for f in F]
                ce = [f[:, 2].min() - k_star for f in F]
                igd = [igd_plus((f - ideal) / span, ref_n) for f in F] if system == '30' else None
                entry = {'evaluations': rows[0]['nfe'], 'cost_gap_pct_runs': cg, 'emission_gap_pct_runs': eg,
                         'curtailment_excess_runs': ce, 'igd_plus_runs': igd,
                         'cost_gap_pct_median': float(np.median(cg)), 'emission_gap_pct_median': float(np.median(eg)),
                         'curtailment_excess_median': float(np.median(ce)),
                         'igd_plus_median': float(np.median(igd)) if igd else None,
                         'time_median': float(np.median([x['time'] for x in rows]))}
                if k == 1:
                    same = [np.array(saved[f"{group}|{alg}"][r]['front']).shape == F[i].shape
                            and np.allclose(saved[f"{group}|{alg}"][r]['front'], F[i], rtol=0, atol=0)
                            for i, r in enumerate(RUNS)]
                    entry['reproduces_unified_runs'] = f"{sum(same)}/{len(same)}"
                summary[f"{system}|{group}|{alg}|x{k}"] = entry
                igd_s = f"{entry['igd_plus_median']:8.3f}" if igd else f"{'-':>8}"
                rep = f"  x1 reproduces saved runs {entry['reproduces_unified_runs']}" if k == 1 else ''
                print(f"{system + '-bus ' + group:<30}{alg:<9}{'x' + str(k):>7}{entry['evaluations']:>13,}"
                      f"{entry['cost_gap_pct_median']:9.2f}%{entry['emission_gap_pct_median']:9.2f}%"
                      f"{entry['curtailment_excess_median']:8.1f}{igd_s}{rep}")
            print()
    return summary


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, (os.cpu_count() or 2) - 1)
    data = load('budget_test.json')['runs'] if os.path.exists(OUT) else {}
    jobs = [(s, k, sc, m, a, r) for s, sc, m, fs in CASES for k in fs for a in ALGORITHMS for r in RUNS]
    todo = [j for j in jobs if job_key(*j) not in data]
    todo.sort(key=lambda j: -j[1] * COST_WEIGHT[j[0]] * (COST_WEIGHT['MOEA/D'] if j[4] == 'MOEA/D' else 1))
    t0 = time.time()
    print(f"{len(jobs) - len(todo)} of {len(jobs)} runs already done; running {len(todo)}", flush=True)
    if todo:
        with Pool(n_workers) as pool:
            for i, (key, res) in enumerate(pool.imap_unordered(run_job, todo), 1):
                data[key] = res
                if i % 5 == 0 or i == len(todo):
                    tmp = OUT + '.tmp'
                    with open(tmp, 'w') as f:
                        json.dump({'runs': data}, f)
                    os.replace(tmp, OUT)
                    print(f"  {i}/{len(todo)} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    summary = analyse(data)
    with open(OUT, 'w') as f:
        json.dump({'cases': [list(c[:3]) + [list(c[3])] for c in CASES], 'seeds_runs': list(RUNS),
                   'summary': summary, 'runs': data}, f)
    print(f"Saved {OUT}")


if __name__ == '__main__':
    main()
