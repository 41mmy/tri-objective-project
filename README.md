# Tri-Objective Project: Economic Load Dispatch with Renewable Energy Penetration

Simulation code and results.

A comparison of four multi-objective algorithms on a 24-hour economic load dispatch problem with high renewable penetration, on two test systems, measured against the exact optimum of the model.

## The problem

A 24-hour dispatch of thermal generators with three objectives:

- fuel cost ($/day),
- CO2 emission (t/day),
- renewable curtailment (MWh/day),

subject to generator limits, hourly power balance, ramp-rate limits and a 10% spinning reserve.

| | IEEE 30-bus | NREL-118 |
|---|---|---|
| Thermal units | 6 | 54 |
| Decision variables | 144 | 1,296 |
| Population / generations | 80 / 200 | 100 / 250 |
| Evaluations per run | 16,080 | 25,100 |

Scenarios: renewable capacity equal to 30%, 50% and 70% of peak demand (60% wind, 40% solar), each in a Deterministic mode and a Robust mode (the 6 highest-renewable hours derated by 15%). Every configuration is run 30 times.

## Algorithms

HADE-NS, NSGA-II, MOPSO and MOEA/D. Each has one implementation, used unchanged on both systems (`simulation/moeld_algorithms.py`):

- **HADE-NS**: DE/rand/1/bin with JADE-style adaptation of F and CR (per generation, from the trials that survive selection), nondominated sorting with crowding, and a local search every 15 generations.
- **NSGA-II**: binary tournament, SBX, polynomial mutation, duplicate elimination.
- **MOPSO**: c1 = c2 = 2.0, inertia 0.9 to 0.4, velocity clamp, crowding-based leader selection, decaying mutation.
- **MOEA/D**: MOEA/D-DE with normalised Tchebycheff and the full simplex-lattice weight set.

The comparison rules are the same for all four:

- every candidate is repaired before evaluation (ramp-aware power-balance repair);
- a solution is feasible when its total constraint violation is at most 0.01;
- every evaluated feasible solution enters an external archive of unique nondominated points, capped at twice the population size;
- run `r` uses the same seed for every algorithm: `seed = r * 1000 + penetration * 10 + (1 if Robust else 0)`;
- the evaluation budget is equal, with two small differences: HADE-NS spends 52 (30-bus) or 80 (NREL-118) extra evaluations on its local search, and MOEA/D uses 16,068 and 25,095 because of its weight count.

## Reference and metrics

The model is convex (quadratic cost and emission, linear constraints), so its exact optimum is computed with a QP solver (`scripts/gap_to_optimum.py`), and the exact IEEE 30-bus Pareto front with the epsilon-constraint method (`scripts/exact_front_igd_30.py`).

- **Gap to the optimum**: each run's lowest cost and lowest emission relative to the exact minimum, and its lowest curtailment above the exact floor.
- **Hypervolume (HV)**: exact, in a fixed frame per scenario and mode: the exact optimum as ideal point, the worst point of the runs as nadir, reference point 1.1. HV values are comparable within one scenario and mode only.
- **IGD+**: distance to the exact Pareto front (IEEE 30-bus only; the NREL-118 front is essentially a single point).
- **Statistics**: two-sided Mann-Whitney U with Holm correction within each scenario and mode, and Cliff's delta.

## Results

All 1,440 runs produce feasible solutions.

| | IEEE 30-bus | NREL-118 |
|---|---|---|
| HV ranking (all six groups) | MOPSO > MOEA/D > HADE-NS > NSGA-II | MOPSO > HADE-NS > MOEA/D > NSGA-II |
| Median cost gap to the optimum | 7.1-10.1% | 1.8-8.2% |
| Median emission gap to the optimum | 2.1-9.1% | 1.5-4.8% |

- No single algorithm is best on the IEEE 30-bus system: MOPSO leads on HV and on emission, the three leading algorithms are often statistically tied on IGD+, and HADE-NS has the lowest median cost at 50% and 70% penetration (significantly so only at 70%).
- On NREL-118 the order is the same for HV, lowest cost and lowest emission, and every pairwise difference is significant.
- No algorithm reaches the exact optimum within the budget. With 16 times the evaluations (5 seeds), the IEEE 30-bus cost gap is still 4.6-8.2% (`scripts/budget_test.py`).

Figures and tables are in `results/final/`.

## Repository layout

```
simulation/
  system_data_30.py      IEEE 30-bus data and load / wind / solar profiles
  system_data_118.py     NREL-118 data and profiles
  moeld_systems.py       both systems, shared evaluate() and repair()
  moeld_algorithms.py    HADE-NS, NSGA-II, MOPSO, MOEA/D
scripts/
  run_unified.py         run the experiments (720 runs per system)
  gap_to_optimum.py      exact optimum and gaps
  exact_front_igd_30.py  exact IEEE 30-bus front and IGD+
  compare_variants.py    HV, gaps, IGD+, statistical tests
  budget_test.py         gaps with 4x and 16x the evaluations
  make_results_figures.py  figures and tables in results/final/
  make_conference_figures.py  single-column versions of the figures (conf_*.png)
  metrics.py             hypervolume, IGD+, statistics helpers
  generate_block_diagram.py, generate_flowchart.py   method diagrams in figures/
results/
  fronts_<system>_unified_v2.json   final runs (every run's archive)
  fronts_<system>_unified.json      first unified runs (archive of survivors only, cap N)
  gap_to_optimum_<system>.json, igd_plus_30.json, compare_<system>_variants.json, budget_test.json
  final/figures, final/tables
data/                    NREL-118 generator data and fitted coefficients
figures/                 method diagrams
```

## Running

```bash
pip install -r requirements.txt
```

The experiments (about 2 hours for the IEEE 30-bus system and 4 hours for NREL-118 on an 8-core machine; a run can be restarted and continues where it stopped):

```bash
python scripts/run_unified.py 30 7
python scripts/run_unified.py 118 7
```

The analysis, in this order:

```bash
python scripts/gap_to_optimum.py 30
python scripts/gap_to_optimum.py 118
python scripts/exact_front_igd_30.py
python scripts/compare_variants.py 30
python scripts/compare_variants.py 118
python scripts/make_results_figures.py
```

The runs are deterministic: the same seed reproduces a saved run exactly on the same machine and library versions.
