#!/usr/bin/env python3
"""
Compare the result sets of one system: hypervolume, gap to the exact optimum, IGD+, tests.
Usage: python scripts/compare_variants.py [30|118]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from metrics import ALGORITHMS, compare, hv3d, igd_plus, rank

SCENARIOS = ['Low (30%)', 'High (50%)', 'Ultra-High (70%)']
MODES = ['Deterministic', 'Robust']
ALL_VARIANTS = ['unified', 'unified_v2']
STATS_VARIANTS = ('unified', 'unified_v2')
FEAS_TOL = 1e-2
REF = 1.1
REF_SENSITIVITY = [1.05, 1.2]
OBJ_NAMES = ['cost', 'emission', 'curtailment']
FILES = {
    '30': {'unified': 'fronts_30_unified.json', 'unified_v2': 'fronts_30_unified_v2.json'},
    '118': {'unified': 'fronts_118_unified.json', 'unified_v2': 'fronts_118_unified_v2.json'},
}
OPTIMUM_FILE = {'30': 'gap_to_optimum_30.json', '118': 'gap_to_optimum_118.json'}
EXACT_FRONT_FILE = {'30': 'igd_plus_30.json'}


def load(path):
    with open(os.path.join(ROOT, 'results', path)) as f:
        return json.load(f)


def feasible_front(run):
    front = np.array(run['front']).reshape(-1, 3)
    if 'cv' not in run:
        return front
    return front[np.array(run['cv']) <= FEAS_TOL]


def main():
    system = sys.argv[1] if len(sys.argv) > 1 else '30'
    if system not in FILES:
        sys.exit(f"system must be one of {list(FILES)}")
    VARIANTS = [v for v in ALL_VARIANTS if os.path.exists(os.path.join(ROOT, 'results', FILES[system][v]))]
    HEADLINE, PREVIOUS = VARIANTS[-1], VARIANTS[0]
    runs = {v: load(FILES[system][v]) for v in VARIANTS}
    cv_available = {v: all('cv' in r for rs in runs[v].values() for r in rs) for v in VARIANTS}
    optimum = load(OPTIMUM_FILE[system])['groups']
    exact = load(EXACT_FRONT_FILE[system])['groups'] if system in EXACT_FRONT_FILE else None

    out = {'system': system, 'variants': VARIANTS, 'headline': HEADLINE,
           'files': {k: v for k, v in FILES[system].items() if k in VARIANTS},
           'cv_available': cv_available,
           'optimum_file': OPTIMUM_FILE[system], 'exact_front_file': EXACT_FRONT_FILE.get(system),
           'method': {
               'feasibility': f'cv <= {FEAS_TOL}',
               'hv': (f'exact, feasible points only; frame per group: ideal = exact optimum (min cost, '
                      f'min emission, curtailment floor), nadir = worst feasible point of {HEADLINE}'),
               'reference_point': REF, 'hv_max': REF ** 3,
               'reference_point_sensitivity': REF_SENSITIVITY,
               'gap': 'per run: (lowest feasible objective - exact minimum) / exact minimum; curtailment: minus exact floor',
               'igd_plus': 'IEEE 30-bus only, normalisation as in igd_plus_30.json',
               'tests': f'{", ".join(STATS_VARIANTS)}: two-sided Mann-Whitney U, Holm within group; Cliff delta',
           }, 'groups': {}}

    for scenario in SCENARIOS:
        for mode in MODES:
            group = f"{scenario}|{mode}"
            opt = optimum[group]['optimum']
            c_star, e_star = opt['min_cost']['cost'], opt['min_emission']['emission']
            k_star = opt['min_curtailment']['curtailment']
            if exact is not None:
                ref_front = np.array(exact[group]['exact_front'])
                ref_ideal = ref_front.min(axis=0)
                ref_span = np.array(exact[group]['normalisation_span'])
                ref_norm = (ref_front - ref_ideal) / ref_span

            feas = {v: {a: [feasible_front(r) for r in runs[v][f"{group}|{a}"]] for a in ALGORITHMS}
                    for v in VARIANTS}
            # HV frame: exact optimum as ideal, headline runs as nadir (older runs must not stretch it)
            head = np.vstack([fr for a in ALGORITHMS for fr in feas[HEADLINE][a] if len(fr)])
            ideal, nadir = np.array([c_star, e_star, k_star]), head.max(axis=0)
            span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)

            g = {'ideal': ideal.tolist(), 'nadir': nadir.tolist(),
                 'exact_optimum': {'min_cost': c_star, 'min_emission': e_star, 'curtailment_floor': k_star},
                 'variants': {}}
            for v in VARIANTS:
                gv, hv, minobj, igd, gaps = {}, {}, {}, {}, {}
                hv_sens = {s: {} for s in REF_SENSITIVITY}
                for a in ALGORITHMS:
                    all_runs = runs[v][f"{group}|{a}"]
                    fr_list = feas[v][a]
                    hv[a] = [hv3d((fr - ideal) / span, np.full(3, REF)) if len(fr) else 0.0 for fr in fr_list]
                    for s in REF_SENSITIVITY:
                        hv_sens[s][a] = float(np.median(
                            [hv3d((fr - ideal) / span, np.full(3, s)) if len(fr) else 0.0 for fr in fr_list]))
                    minobj[a] = np.array([fr.min(axis=0) if len(fr) else np.full(3, np.nan) for fr in fr_list])
                    n_pts = sum(len(r['front']) for r in all_runs)
                    gaps[a] = {'cost_gap_pct': 100 * (minobj[a][:, 0] / c_star - 1),
                               'emission_gap_pct': 100 * (minobj[a][:, 1] / e_star - 1),
                               'curtailment_excess': minobj[a][:, 2] - k_star}
                    gv[a] = {
                        'runs_with_feasible': int(sum(len(fr) > 0 for fr in fr_list)),
                        'feasible_point_share': float(sum(len(fr) for fr in fr_list) / max(n_pts, 1)),
                        'cv_median_all_points': (
                            float(np.median(np.concatenate([r['cv'] for r in all_runs])))
                            if cv_available[v] and n_pts else None),
                        'hv_runs': hv[a],
                        'hv_median': float(np.median(hv[a])),
                        'hv_q1': float(np.percentile(hv[a], 25)), 'hv_q3': float(np.percentile(hv[a], 75)),
                        'unique_feasible_archive_median': float(np.median(
                            [len(np.unique(np.round(fr, 6), axis=0)) for fr in fr_list])),
                        'time_median': float(np.median([r['time'] for r in all_runs])),
                    }
                    for j, name in enumerate(OBJ_NAMES):
                        col = minobj[a][:, j]
                        gv[a][f'min_{name}_runs'] = [None if np.isnan(x) else float(x) for x in col]
                        gv[a][f'min_{name}_median_feasible'] = (
                            float(np.nanmedian(col)) if np.any(~np.isnan(col)) else None)
                    for name, vals in gaps[a].items():
                        gv[a][f'{name}_runs'] = [None if np.isnan(x) else float(x) for x in vals]
                        gv[a][f'{name}_median'] = float(np.nanmedian(vals)) if np.any(~np.isnan(vals)) else None
                    if exact is not None:
                        igd[a] = [igd_plus((fr - ref_ideal) / ref_span, ref_norm) if len(fr) else np.nan
                                  for fr in fr_list]
                        gv[a]['igd_plus_runs'] = [None if np.isnan(x) else float(x) for x in igd[a]]
                        gv[a]['igd_plus_median'] = (float(np.nanmedian(igd[a]))
                                                    if np.any(~np.isnan(igd[a])) else None)
                rank_main = rank({a: gv[a]['hv_median'] for a in ALGORITHMS}, True)
                rank_sens = {str(s): rank(hv_sens[s], True) for s in REF_SENSITIVITY}
                g['variants'][v] = {
                    'algorithms': gv,
                    'rank_hv': rank_main,
                    'hv_median_ref_sensitivity': {str(s): hv_sens[s] for s in REF_SENSITIVITY},
                    'rank_hv_ref_sensitivity': rank_sens,
                    'rank_hv_stable_across_refs': all(r == rank_main for r in rank_sens.values()),
                }
                if v in STATS_VARIANTS:
                    gs = g['variants'][v]
                    gs['stats_hv'] = compare(hv, True)
                    for j, name in enumerate(OBJ_NAMES):
                        vals = {a: minobj[a][:, j] for a in ALGORITHMS}
                        if all(not np.any(np.isnan(x)) for x in vals.values()):
                            gs[f'rank_min_{name}'] = rank({a: float(np.median(x)) for a, x in vals.items()}, False)
                            gs[f'stats_min_{name}'] = compare(vals, False)
                    if exact is not None and all(not np.any(np.isnan(igd[a])) for a in ALGORITHMS):
                        gs['rank_igd_plus'] = rank({a: float(np.median(igd[a])) for a in ALGORITHMS}, False)
                        gs['stats_igd_plus'] = compare({a: np.array(igd[a]) for a in ALGORITHMS}, False)

            out['groups'][group] = g

    out_name = f'compare_{system}_variants.json'
    with open(os.path.join(ROOT, 'results', out_name), 'w') as f:
        json.dump(out, f, indent=1)

    fmt = lambda x, w=12: f"{x:>{w},.0f}" if x is not None else f"{'-':>{w}}"
    pct = lambda x: f"{x:>7.1f}%" if x is not None else f"{'-':>8}"
    for group, g in out['groups'].items():
        e = g['exact_optimum']
        print(f"\n=== {group} ===   exact optimum: cost {e['min_cost']:,.0f}, emission {e['min_emission']:,.3f}, "
              f"curtailment floor {e['curtailment_floor']:,.1f}")
        igd_hdr = f"{'IGD+':>8}" if exact is not None else ''
        print(f"{'Algorithm':<9} |{PREVIOUS:>28} |{HEADLINE + ' (headline)':>57}{' ' * len(igd_hdr)} |")
        print(f"{'':<9} |{'min cost':>12}{'HV':>9}{'uniq':>6}  |"
              f"{'feas':>6}{'min cost':>12}{'cost gap':>9}{'emis gap':>9}{'curt+':>8}{'HV':>8}{'uniq':>5}{igd_hdr} |")
        for a in ALGORITHMS:
            f_, u = g['variants'][PREVIOUS]['algorithms'][a], g['variants'][HEADLINE]['algorithms'][a]
            igd_col = f"{u['igd_plus_median']:>8.3f}" if exact is not None and u.get('igd_plus_median') is not None else (
                f"{'-':>8}" if exact is not None else '')
            print(f"{a:<9} |"
                  f"{fmt(f_['min_cost_median_feasible'])}{f_['hv_median']:>9.4f}{f_['unique_feasible_archive_median']:>6.0f}  |"
                  f"{u['runs_with_feasible']:>3}/30{fmt(u['min_cost_median_feasible'])}{pct(u['cost_gap_pct_median'])}"
                  f"{pct(u['emission_gap_pct_median'])}{u['curtailment_excess_median']:>8.1f}{u['hv_median']:>8.4f}"
                  f"{u['unique_feasible_archive_median']:>5.0f}{igd_col} |")
        hl = g['variants'][HEADLINE]
        print(f"  {PREVIOUS:<10} rank (common HV):   {g['variants'][PREVIOUS]['rank_hv']}")
        print(f"  {HEADLINE:<10} rank (common HV):   {hl['rank_hv']}")
        if 'rank_min_cost' in hl:
            print(f"  {HEADLINE:<10} rank (min cost):    {hl['rank_min_cost']}")
        if 'rank_igd_plus' in hl:
            print(f"  {HEADLINE:<10} rank (IGD+):        {hl['rank_igd_plus']}")
        unstable = [v for v in VARIANTS if not g['variants'][v]['rank_hv_stable_across_refs']]
        print(f"  HV ranks same at ref {REF_SENSITIVITY[0]}/{REF}/{REF_SENSITIVITY[1]}: "
              + ('yes, all variants' if not unstable else f"NO for {unstable}"))
        ns = [k for k, s in hl['stats_hv'].items() if not s['sig_holm']]
        print(f"  {HEADLINE} HV pairs NOT significant after Holm: {ns if ns else 'none'}")
        if 'stats_igd_plus' in hl:
            ns = [k for k, s in hl['stats_igd_plus'].items() if not s['sig_holm']]
            print(f"  {HEADLINE} IGD+ pairs NOT significant after Holm: {ns if ns else 'none'}")
    if not all(cv_available.values()):
        print(f"\nNote: no constraint violation saved for {[v for v, ok in cv_available.items() if not ok]}"
              f" -> all archive points treated as feasible")
    print(f"\nSaved results/{out_name}")


if __name__ == '__main__':
    main()
