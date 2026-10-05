#!/usr/bin/env python3
"""
Final figures and tables from the newest unified runs, written to results/final/.
Usage: python scripts/make_results_figures.py
"""
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
OUT_FIG = os.path.join(RES, 'final', 'figures')
OUT_TAB = os.path.join(RES, 'final', 'tables')

ALGORITHMS = ['HADE-NS', 'NSGA-II', 'MOPSO', 'MOEA/D']
SCENARIOS = ['Low (30%)', 'High (50%)', 'Ultra-High (70%)']
SHORT = {'Low (30%)': '30%', 'High (50%)': '50%', 'Ultra-High (70%)': '70%'}
MODES = ['Deterministic', 'Robust']
SYSTEM_NAME = {'30': 'IEEE 30-bus', '118': 'NREL-118'}
HEADLINE = 'unified'

COLOR = {'HADE-NS': '#2a78d6', 'NSGA-II': '#eb6834', 'MOPSO': '#1baf7a', 'MOEA/D': '#eda100'}
MARKER = {'HADE-NS': 'o', 'NSGA-II': 's', 'MOPSO': '^', 'MOEA/D': 'D'}
SURFACE, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Arial'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': BASE, 'axes.linewidth': 0.8, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.6,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'legend.frameon': False, 'legend.fontsize': 9, 'text.color': INK,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def med_iqr(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    return np.median(v), np.percentile(v, 25), np.percentile(v, 75)


def legend_handles(kind='patch'):
    if kind == 'patch':
        return [Patch(facecolor=COLOR[a], edgecolor=COLOR[a], label=a) for a in ALGORITHMS]
    return [Line2D([], [], marker=MARKER[a], color=COLOR[a], linestyle='none', markersize=7,
                   markeredgecolor=SURFACE, label=a) for a in ALGORITHMS]


def finish(fig, title, handles, path):
    """Legend only: the descriptive title lives in the paper caption, so the image carries none."""
    w, h = fig.get_size_inches()
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), bbox_to_anchor=(0.5, 1.0),
               handlelength=1.4, columnspacing=2.0, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / h))
    fig.savefig(path)
    plt.close(fig)


def box_panel(ax, data, ylabel=None):
    bp = ax.boxplot([data[a] for a in ALGORITHMS], widths=0.6, patch_artist=True,
                    medianprops={'color': INK, 'linewidth': 2},
                    whiskerprops={'linewidth': 1}, capprops={'linewidth': 1},
                    flierprops={'marker': 'o', 'markersize': 3, 'markeredgewidth': 0})
    for i, a in enumerate(ALGORITHMS):
        bp['boxes'][i].set(facecolor=COLOR[a] + '59', edgecolor=COLOR[a], linewidth=1.5)
        for part in ('whiskers', 'caps'):
            for line in bp[part][2 * i:2 * i + 2]:
                line.set_color(COLOR[a])
        bp['fliers'][i].set(visible=False)
        v = np.array(data[a], dtype=float)
        jit = np.random.default_rng(i).uniform(-0.2, 0.2, len(v))
        ax.scatter(i + 1 + jit, v, s=9, color=COLOR[a], alpha=0.75, linewidths=0, zorder=3)
        ax.annotate(f"{np.median(v):.3f}", (i + 1, np.median(v)), xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=7.5, color=INK, zorder=4,
                    bbox={'boxstyle': 'round,pad=0.12', 'fc': SURFACE, 'ec': 'none', 'alpha': 0.85})
    ax.margins(y=0.12)
    ax.set_xticks(range(1, len(ALGORITHMS) + 1), ALGORITHMS, rotation=0)
    ax.tick_params(axis='x', length=0)
    if ylabel:
        ax.set_ylabel(ylabel)


def grid_of_boxes(comp, metric_key, ylabel, title, path):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    for r, mode in enumerate(MODES):
        for c, sc in enumerate(SCENARIOS):
            algs = comp[f"{sc}|{mode}"]['variants'][HEADLINE]['algorithms']
            data = {a: [x for x in algs[a][metric_key] if x is not None] for a in ALGORITHMS}
            box_panel(axes[r, c], data, ylabel if c == 0 else None)
            axes[r, c].set_title(f"{SHORT[sc]} penetration · {mode}", fontweight='bold')
            axes[r, c].set_xticklabels([] if r == 0 else ALGORITHMS)
    fig.text(0.01, 0.005, 'Dots: individual runs. Box: IQR. Black line and printed value: median. '
             'Y-axis differs per panel.', fontsize=8, color=MUTED)
    finish(fig, title, legend_handles(), path)


def gap_figure(comp, system, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
    x = np.arange(len(SCENARIOS))
    width = 0.2
    rows = [('cost_gap_pct_runs', 'Cost above exact optimum (%)'),
            ('emission_gap_pct_runs', 'Emission above exact optimum (%)')]
    for r, (key, ylabel) in enumerate(rows):
        for c, mode in enumerate(MODES):
            ax = axes[r, c]
            for k, a in enumerate(ALGORITHMS):
                stats = [med_iqr(comp[f"{sc}|{mode}"]['variants'][HEADLINE]['algorithms'][a][key]) for sc in SCENARIOS]
                med = np.array([s[0] for s in stats])
                err = np.array([[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]])
                pos = x + (k - 1.5) * width
                ax.bar(pos, med, width, color=COLOR[a], edgecolor=SURFACE, linewidth=1.5, label=a, zorder=2)
                ax.errorbar(pos, med, yerr=err, fmt='none', ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
                for p_, m_, e_ in zip(pos, med, err[1]):
                    ax.annotate(f"{m_:.1f}", (p_, m_ + e_), xytext=(0, 2), textcoords='offset points',
                                ha='center', fontsize=7.5, color=INK)
            ax.axhline(0, color=BASE, linewidth=0.8, zorder=1)
            ax.set_title(mode, fontweight='bold')
            ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
            ax.set_xticks(x, [f"{SHORT[sc]} penetration" for sc in SCENARIOS])
            ax.tick_params(axis='x', length=0)
            if c == 0:
                ax.set_ylabel(ylabel)
    finish(fig, f"{SYSTEM_NAME[system]}: median gap of each run's best solution to the exact optimum "
                f"(whiskers: interquartile range, 30 runs)", legend_handles(), path)


def median_run_front(comp, fronts, key, a):
    algs = comp[key]['variants'][HEADLINE]['algorithms']
    igd = np.array([np.inf if v is None else v for v in algs[a]['igd_plus_runs']])
    run = int(np.argsort(igd)[len(igd) // 2])            # median-IGD+ run
    F = np.array(fronts[f"{key}|{a}"][run]['front']).reshape(-1, 3)
    return F[np.argsort(F[:, 0])]


def fronts_figure(comp, fronts, exact, path):
    """Overview: all algorithms in one panel, thin lines so each archive stays traceable."""
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4))
    for r, mode in enumerate(MODES):
        for c, sc in enumerate(SCENARIOS):
            ax, key = axes[r, c], f"{sc}|{mode}"
            ref = np.array(exact[key]['exact_front'])
            ref = ref[np.argsort(ref[:, 0])]
            ax.plot(ref[:, 0], ref[:, 1], color=INK, linewidth=2.2, zorder=5)
            for a in ALGORITHMS:
                F = median_run_front(comp, fronts, key, a)
                ax.plot(F[:, 0], F[:, 1], color=COLOR[a], linewidth=1.1, alpha=0.9, zorder=2)
                ax.scatter(F[:, 0], F[:, 1], s=9, marker=MARKER[a], color=COLOR[a], linewidths=0, zorder=3)
            ax.set_title(f"{SHORT[sc]} penetration · {mode}", fontweight='bold')
            ax.grid(True, axis='both')
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            if r == 1:
                ax.set_xlabel('Fuel cost ($/day)')
            if c == 0:
                ax.set_ylabel('CO₂ emission (t/day)')
    handles = [Line2D([], [], color=INK, linewidth=2.2, label='Exact Pareto front')] + legend_handles('marker')
    finish(fig, "IEEE 30-bus: median-IGD+ run of each algorithm against the exact Pareto front "
                "(closer to the black curve is better)", handles, path)


def fronts_by_algorithm(comp, fronts, exact, mode, path):
    """One row per algorithm, one column per penetration level: no overplotting."""
    fig, axes = plt.subplots(len(ALGORITHMS), 3, figsize=(10.5, 10), sharex='col', sharey='col')
    for c, sc in enumerate(SCENARIOS):
        key = f"{sc}|{mode}"
        ref = np.array(exact[key]['exact_front'])
        ref = ref[np.argsort(ref[:, 0])]
        for r, a in enumerate(ALGORITHMS):
            ax = axes[r, c]
            ax.plot(ref[:, 0], ref[:, 1], color=INK, linewidth=2, zorder=1)
            F = median_run_front(comp, fronts, key, a)
            ax.plot(F[:, 0], F[:, 1], color=COLOR[a], linewidth=1, alpha=0.7, zorder=2)
            ax.scatter(F[:, 0], F[:, 1], s=14, marker=MARKER[a], color=COLOR[a], edgecolors='none',
                       linewidths=0, zorder=3)
            ax.grid(True, axis='both')
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            if r == 0:
                ax.set_title(f"{SHORT[sc]} penetration", fontweight='bold')
            if c == 0:
                ax.set_ylabel(f"{a}\nCO₂ emission (t/day)", color=COLOR[a], fontweight='bold')
            if r == len(ALGORITHMS) - 1:
                ax.set_xlabel('Fuel cost ($/day)')
    handles = [Line2D([], [], color=INK, linewidth=2, label='Exact Pareto front')] + legend_handles('marker')
    finish(fig, f"IEEE 30-bus, {mode.lower()}: median-IGD+ archive of each algorithm vs the exact front",
           handles, path)


def write_table(rows, header, base):
    with open(base + '.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with open(base + '.md', 'w', encoding='utf-8') as f:
        f.write('| ' + ' | '.join(header) + ' |\n')
        f.write('|' + '|'.join(['---'] * len(header)) + '|\n')
        for row in rows:
            f.write('| ' + ' | '.join(str(v) for v in row) + ' |\n')


def tables(comp, fronts, system):
    igd = system == '30'
    header = ['Scenario', 'Mode', 'Algorithm', 'Runs feasible', 'Evaluations',
              'Lowest cost (median)', 'Cost gap % (median)', 'Emission gap % (median)',
              'Curtailment above floor MWh (median)', 'HV median', 'HV IQR'] + \
             (['IGD+ median'] if igd else []) + ['Archive size (median)']
    rows = []
    for sc in SCENARIOS:
        for mode in MODES:
            g = comp[f"{sc}|{mode}"]
            for a in ALGORITHMS:
                u = g['variants'][HEADLINE]['algorithms'][a]
                nfe = sorted({r['nfe'] for r in fronts[f"{sc}|{mode}|{a}"]})
                row = [SHORT[sc], mode, a, f"{u['runs_with_feasible']}/30", '/'.join(f"{n:,}" for n in nfe),
                       f"{u['min_cost_median_feasible']:,.1f}", f"{u['cost_gap_pct_median']:.2f}",
                       f"{u['emission_gap_pct_median']:.2f}", f"{u['curtailment_excess_median']:.1f}",
                       f"{u['hv_median']:.4f}", f"[{u['hv_q1']:.4f}, {u['hv_q3']:.4f}]"]
                if igd:
                    row.append(f"{u['igd_plus_median']:.4f}")
                row.append(f"{u['unique_feasible_archive_median']:.0f}")
                rows.append(row)
    write_table(rows, header, os.path.join(OUT_TAB, f'summary_{system}'))

    metrics = [('stats_hv', 'HV (higher better)'), ('stats_min_cost', 'Lowest cost (lower better)')]
    if igd:
        metrics.append(('stats_igd_plus', 'IGD+ (lower better)'))
    header = ['Scenario', 'Mode', 'Metric', 'Comparison', 'p (Holm)', 'Significant', "Cliff's delta", 'Magnitude', 'Favours']
    rows = []
    for sc in SCENARIOS:
        for mode in MODES:
            hl = comp[f"{sc}|{mode}"]['variants'][HEADLINE]
            for key, label in metrics:
                for pair, s in hl.get(key, {}).items():
                    rows.append([SHORT[sc], mode, label, pair, f"{s['p_holm']:.2e}", 'yes' if s['sig_holm'] else 'no',
                                 f"{s['cliffs_delta']:+.3f}", s['magnitude'], s['favors']])
    write_table(rows, header, os.path.join(OUT_TAB, f'tests_{system}'))


def budget_outputs():
    bt = load('budget_test.json')
    cases, summary = bt['cases'], bt['summary']
    fig, axes = plt.subplots(2, len(cases), figsize=(3.6 * len(cases), 6), squeeze=False)
    rows_def = [('cost_gap_pct_runs', 'Cost above exact optimum (%)'),
                ('emission_gap_pct_runs', 'Emission above exact optimum (%)')]
    table = []
    for c, (system, scenario, mode, factors) in enumerate(cases):
        for r, (key, ylabel) in enumerate(rows_def):
            ax = axes[r, c]
            for a in ALGORITHMS:
                ent = [summary[f"{system}|{scenario}|{mode}|{a}|x{k}"] for k in factors]
                x = [e['evaluations'] for e in ent]
                med = [np.median(e[key]) for e in ent]
                ax.plot(x, med, color=COLOR[a], linewidth=2, marker=MARKER[a], markersize=6,
                        markeredgecolor=SURFACE, label=a)
            ax.set_xscale('log')
            ax.set_xticks(x, [f"{v / 1000:,.0f}k" for v in x])
            ax.minorticks_off()
            ax.set_ylim(bottom=0)
            ax.grid(True, axis='both')
            if r == 0:
                ax.set_title(f"{SYSTEM_NAME[system]} · {SHORT[scenario]} · {mode}", fontweight='bold')
            else:
                ax.set_xlabel('Function evaluations (log scale)')
            if c == 0:
                ax.set_ylabel(ylabel)
        for a in ALGORITHMS:
            for k in factors:
                e = summary[f"{system}|{scenario}|{mode}|{a}|x{k}"]
                table.append([SYSTEM_NAME[system], SHORT[scenario], mode, a, f"x{k}", f"{e['evaluations']:,}",
                              f"{e['cost_gap_pct_median']:.2f}", f"{e['emission_gap_pct_median']:.2f}",
                              f"{e['curtailment_excess_median']:.1f}",
                              f"{e['igd_plus_median']:.4f}" if e['igd_plus_median'] is not None else '-'])
    n_seeds = len(bt['seeds_runs'])
    finish(fig, f"Gap to the exact optimum against the evaluation budget (median of {n_seeds} seeds; "
                f"x1 = the paper's budget)", legend_handles('marker'), os.path.join(OUT_FIG, 'budget_test.png'))
    write_table(table, ['System', 'Scenario', 'Mode', 'Algorithm', 'Budget', 'Evaluations',
                        'Cost gap % (median)', 'Emission gap % (median)', 'Curtailment above floor MWh (median)',
                        'IGD+ median'], os.path.join(OUT_TAB, 'budget_test'))


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_TAB, exist_ok=True)
    exact = load('igd_plus_30.json')['groups']
    for system in ('30', '118'):
        global HEADLINE
        full = load(f'compare_{system}_variants.json')
        HEADLINE = full.get('headline', 'unified')
        comp = full['groups']
        fronts = load(full['files'][HEADLINE])
        print(f"{SYSTEM_NAME[system]}: figures and tables from '{HEADLINE}' ({full['files'][HEADLINE]})")
        grid_of_boxes(comp, 'hv_runs', 'Hypervolume (normalised, ref. 1.1)',
                      f"{SYSTEM_NAME[system]}: hypervolume per run (30 runs; higher is better; "
                      f"comparable within a panel only)", os.path.join(OUT_FIG, f'hv_box_{system}.png'))
        gap_figure(comp, system, os.path.join(OUT_FIG, f'gap_{system}.png'))
        tables(comp, fronts, system)
        if system == '30':
            grid_of_boxes(comp, 'igd_plus_runs', 'IGD+ to the exact front',
                          "IEEE 30-bus: IGD+ per run (30 runs; lower is better)",
                          os.path.join(OUT_FIG, 'igd_box_30.png'))
            fronts_figure(comp, fronts, exact, os.path.join(OUT_FIG, 'fronts_30.png'))
            for mode in MODES:
                fronts_by_algorithm(comp, fronts, exact, mode,
                                    os.path.join(OUT_FIG, f'fronts_30_by_algorithm_{mode.lower()}.png'))
    if os.path.exists(os.path.join(RES, 'budget_test.json')):
        if 'summary' in load('budget_test.json'):
            budget_outputs()
        else:
            print("budget_test.json is incomplete (budget test still running?) - budget figure skipped")
    for d in (OUT_FIG, OUT_TAB):
        for name in sorted(os.listdir(d)):
            print(os.path.relpath(os.path.join(d, name), ROOT))


if __name__ == '__main__':
    main()
