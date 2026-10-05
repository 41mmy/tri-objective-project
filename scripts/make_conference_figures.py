#!/usr/bin/env python3
"""
Single-column (3.5 in) figures for the 6-page conference paper, written to results/final/figures/.
Usage: python scripts/make_conference_figures.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import make_results_figures as base
from make_results_figures import ALGORITHMS, COLOR, MARKER, SHORT, SYSTEM_NAME, OUT_FIG, load

sys.path.insert(0, os.path.join(base.ROOT, 'simulation'))
from moeld_systems import SYSTEMS

INK, GRID = '#000000', '#d9d9d9'
COL_W = 3.5

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'legend.fontsize': 7, 'axes.facecolor': 'white', 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.edgecolor': INK, 'axes.linewidth': 0.5, 'axes.labelcolor': INK, 'axes.titlecolor': INK,
    'xtick.color': INK, 'ytick.color': INK, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2,
    'grid.color': GRID, 'grid.linewidth': 0.4, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})


def markers():
    return [Line2D([], [], marker=MARKER[a], color=COLOR[a], linestyle='none', markersize=4, label=a)
            for a in ALGORITHMS]


def save(fig, handles, name, top=0.24):
    h = fig.get_size_inches()[1]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), bbox_to_anchor=(0.5, 1.0),
               handlelength=1.2, handletextpad=0.3, columnspacing=0.9, borderpad=0)
    fig.tight_layout(rect=(0, 0, 1, 1 - top / h), pad=0.3, w_pad=0.8, h_pad=0.6)
    path = os.path.join(OUT_FIG, name)
    fig.savefig(path)
    plt.close(fig)
    print(os.path.relpath(path, base.ROOT))


def fronts(comp, fronts_, exact):
    """Median-IGD+ archive of each algorithm against the exact front, deterministic 30% and 70%."""
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.95))
    for ax, sc in zip(axes, ['Low (30%)', 'Ultra-High (70%)']):
        key = f"{sc}|Deterministic"
        ref = np.array(exact[key]['exact_front'])
        ref = ref[np.argsort(ref[:, 0])]
        ax.plot(ref[:, 0] / 1000, ref[:, 1], color=INK, linewidth=1.3, zorder=5)
        for a in ALGORITHMS:
            F = base.median_run_front(comp, fronts_, key, a)
            ax.plot(F[:, 0] / 1000, F[:, 1], color=COLOR[a], linewidth=0.5, zorder=2)
            ax.scatter(F[:, 0] / 1000, F[:, 1], s=3, marker=MARKER[a], color=COLOR[a], linewidths=0, zorder=3)
        ax.set_title(f"{SHORT[sc]} penetration")
        ax.set_xlabel('Fuel cost (10$^3$ \\$/day)')
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    axes[0].set_ylabel('CO$_2$ emission (t/day)')
    handles = [Line2D([], [], color=INK, linewidth=1.3, label='Exact')] + markers()
    save(fig, handles, 'conf_fronts_30.png')


def hv_boxes(comps):
    """Hypervolume of the 30 runs, deterministic mode: one row per system, one column per penetration."""
    fig, axes = plt.subplots(2, 3, figsize=(COL_W, 2.9))
    for r, system in enumerate(('30', '118')):
        for c, sc in enumerate(base.SCENARIOS):
            ax = axes[r, c]
            algs = comps[system][0][f"{sc}|Deterministic"]['variants'][comps[system][1]]['algorithms']
            data = [np.array(algs[a]['hv_runs'], dtype=float) for a in ALGORITHMS]
            bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                            medianprops={'color': INK, 'linewidth': 0.9},
                            whiskerprops={'linewidth': 0.5}, capprops={'linewidth': 0.5})
            for i, a in enumerate(ALGORITHMS):
                bp['boxes'][i].set(facecolor=COLOR[a] + '66', edgecolor=COLOR[a], linewidth=0.7)
                jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(data[i]))
                ax.scatter(i + 1 + jit, data[i], s=1.2, color=COLOR[a], linewidths=0, zorder=3)
            ax.set_xticks([])
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            if r == 0:
                ax.set_title(f"{SHORT[sc]} penetration")
            if c == 0:
                ax.set_ylabel(f"{SYSTEM_NAME[system]}\nhypervolume")
    handles = [Line2D([], [], marker='s', color=COLOR[a], linestyle='none', markersize=4, label=a)
               for a in ALGORITHMS]
    save(fig, handles, 'conf_hv_box.png')


def budget():
    """Median cost gap to the exact optimum against the evaluation budget."""
    bt = load('budget_test.json')
    cases, summary = bt['cases'], bt['summary']
    fig, axes = plt.subplots(1, len(cases), figsize=(COL_W, 1.75), squeeze=False)
    for ax, (system, scenario, mode, factors) in zip(axes[0], cases):
        for a in ALGORITHMS:
            ent = [summary[f"{system}|{scenario}|{mode}|{a}|x{k}"] for k in factors]
            x = [e['evaluations'] for e in ent]
            ax.plot(x, [e['cost_gap_pct_median'] for e in ent], color=COLOR[a], linewidth=0.9,
                    marker=MARKER[a], markersize=3)
        ax.set_xscale('log')
        ax.set_xticks(x, [f"{v / 1000:,.0f}k" for v in x])
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.grid(True, axis='y')
        ax.set_title(f"{SYSTEM_NAME[system]}, {SHORT[scenario]}")
        ax.set_xlabel('Evaluations')
    axes[0, 0].set_ylabel('Cost gap to optimum (%)')
    save(fig, markers(), 'conf_budget.png')


def floor():
    """Why curtailment has a floor: renewable availability against the most the fleet can absorb, 70%."""
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.85))
    hours = np.arange(24)
    for ax, key, unit, div in zip(axes, ('30', '118'), ('MW', 'GW'), (1, 1000)):
        S = SYSTEMS[key]
        D, R = S.demand() / div, S.re_avail(70) / div
        room = D - S.TOTAL_PMIN / div
        ax.step(hours, D, where='mid', color=INK, linewidth=0.9)
        ax.step(hours, room, where='mid', color=INK, linewidth=0.9, linestyle='--')
        ax.step(hours, R, where='mid', color=COLOR['MOPSO'], linewidth=0.9)
        ax.fill_between(hours, room, R, where=R > room, step='mid', color=COLOR['NSGA-II'], alpha=0.6,
                        linewidth=0)
        ax.set_title(f"{SYSTEM_NAME[key]}, 70%")
        ax.set_xlabel('Hour')
        ax.set_ylabel(f"Power ({unit})")
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_ylim(bottom=0)
    handles = [Line2D([], [], color=INK, linewidth=0.9, label='Demand'),
               Line2D([], [], color=INK, linewidth=0.9, linestyle='--', label='Demand − min. thermal'),
               Line2D([], [], color=COLOR['MOPSO'], linewidth=0.9, label='Renewable'),
               Line2D([], [], marker='s', color=COLOR['NSGA-II'], alpha=0.6, linestyle='none', markersize=4,
                      label='Floor')]
    save(fig, handles, 'conf_floor.png')


def main():
    exact = load('igd_plus_30.json')['groups']
    comps = {}
    for system in ('30', '118'):
        full = load(f'compare_{system}_variants.json')
        comps[system] = (full['groups'], full.get('headline', 'unified'))
    base.HEADLINE = comps['30'][1]
    full30 = load('compare_30_variants.json')
    fronts(comps['30'][0], load(full30['files'][base.HEADLINE]), exact)
    hv_boxes(comps)
    budget()
    floor()


if __name__ == '__main__':
    main()
