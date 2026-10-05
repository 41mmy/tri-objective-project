#!/usr/bin/env python3
"""Generate MO-ELD Methodology Block Diagram for IEEE paper."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(7.5, 9.0), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Color scheme - professional IEEE style
c_input = '#D6EAF8'    # light blue
c_form = '#D5F5E3'     # light green
c_algo = '#FDEBD0'     # light orange
c_robust = '#F9E79F'   # light yellow
c_output = '#E8DAEF'   # light purple
c_stat = '#FADBD8'     # light red
border = '#2C3E50'      # dark border
text_color = '#1B2631'

def draw_box(x, y, w, h, title, items, color, title_size=8, item_size=6.5, bold_title=True):
    """Draw a rounded box with title and bullet items."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor=border, linewidth=1.2)
    ax.add_patch(box)
    # Title
    ax.text(x + w/2, y + h - 0.22, title, ha='center', va='top',
            fontsize=title_size, fontweight='bold' if bold_title else 'normal',
            color=text_color, fontfamily='serif')
    # Items
    for i, item in enumerate(items):
        ax.text(x + 0.15, y + h - 0.52 - i*0.24, f"• {item}", ha='left', va='top',
                fontsize=item_size, color=text_color, fontfamily='serif')

def draw_arrow(x1, y1, x2, y2, color='#566573', style='->', lw=1.5):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               connectionstyle='arc3,rad=0'))

def draw_curved_arrow(x1, y1, x2, y2, rad=0.2, color='#566573', lw=1.5):
    """Draw a curved arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle=f'arc3,rad={rad}'))

# ── Title ──
ax.text(5.0, 11.7, 'Proposed MO-ELD Framework', ha='center', va='top',
        fontsize=12, fontweight='bold', color=text_color, fontfamily='serif')

# ═══════════════════════════════════════
# ROW 1: INPUT DATA (y=10.0 to 11.2)
# ═══════════════════════════════════════

# System Data
draw_box(0.2, 10.0, 3.0, 1.2, 'Test System Data',
         ['IEEE 30-bus (6 generators, 144 DV)',
          'NREL-118 bus (54 generators, 1,296 DV)',
          'Generator parameters (Table I)'],
         c_input)

# Load & RE Profiles
draw_box(3.6, 10.0, 3.0, 1.2, 'Demand & RE Profiles',
         ['24-hour load profile',
          '60/40 wind-solar split',
          'Penetration: 30%, 50%, 70%'],
         c_input)

# Uncertainty Model
draw_box(7.0, 10.0, 2.8, 1.2, 'Uncertainty Model',
         ['Bertsimas–Sim selective',
          'Γ = 6 (25% of hours)',
          '±15% forecast error'],
         c_robust)

# ═══════════════════════════════════════
# ROW 2: PROBLEM FORMULATION (y=7.8 to 9.4)
# ═══════════════════════════════════════

draw_box(0.5, 7.7, 4.2, 1.8, 'Tri-Objective Problem Formulation',
         ['f₁: Minimize total fuel cost ($/day)',
          'f₂: Minimize CO₂ emissions (ton/day)',
          'f₃: Minimize renewable curtailment (MWh/day)',
          'Subject to:'],
         c_form)

draw_box(5.2, 7.7, 4.3, 1.8, 'System Constraints',
         ['Power balance: ΣPᵢ,ₜ + Pᵃᶜᶜ = Dₜ',
          'Capacity: Pᵢᵐⁱⁿ ≤ Pᵢ,ₜ ≤ Pᵢᵐᵃˣ',
          'Ramp-rate: |Pᵢ,ₜ − Pᵢ,ₜ₋₁| ≤ Rᵢ',
          'Reserve: Σ(Pᵢᵐᵃˣ − Pᵢ,ₜ) ≥ 10%·Dₜ',
          'RE acceptance: 0 ≤ Pᵃᶜᶜ ≤ Pᵃᵛᵃⁱˡ'],
         c_form, item_size=6.2)

# Arrows from Row 1 to Row 2
draw_arrow(1.7, 10.0, 2.6, 9.5)
draw_arrow(5.1, 10.0, 5.1, 9.5)
draw_arrow(8.4, 10.0, 7.4, 9.5)

# ═══════════════════════════════════════
# ROW 3: OPTIMIZATION ENGINE (y=5.0 to 7.2)
# ═══════════════════════════════════════

# Main box
engine_box = FancyBboxPatch((0.3, 4.8), 9.4, 2.5, boxstyle="round,pad=0.1",
                             facecolor='#FAFAFA', edgecolor=border, linewidth=1.5,
                             linestyle='--')
ax.add_patch(engine_box)
ax.text(5.0, 7.15, 'Multi-Objective Optimization Engine', ha='center', va='top',
        fontsize=9, fontweight='bold', color=text_color, fontfamily='serif')

# Four algorithm boxes
algo_w, algo_h = 2.05, 1.5
algo_y = 5.05

# HADE-NS
draw_box(0.55, algo_y, algo_w, algo_h, 'HADE-NS (Primary)',
         ['DE/rand/1/bin mutation',
          'JADE adaptive F, CR',
          'Nondominated sorting',
          'Local search (every 15 gen)'],
         '#FDEBD0', title_size=7, item_size=5.8)

# NSGA-II
draw_box(2.75, algo_y, algo_w, algo_h, 'NSGA-II',
         ['SBX crossover',
          'Polynomial mutation',
          'Crowding distance',
          'Binary tournament'],
         '#FCE4EC', title_size=7, item_size=5.8)

# MOPSO
draw_box(4.95, algo_y, algo_w, algo_h, 'MOPSO',
         ['Velocity update',
          'Grid-based leader',
          'External archive',
          'Inertia damping'],
         '#E3F2FD', title_size=7, item_size=5.8)

# MOEA/D
draw_box(7.15, algo_y, algo_w, algo_h, 'MOEA/D',
         ['Tchebycheff scalarization',
          'Das-Dennis weights',
          'Neighborhood DE (T=20)',
          'Archive (every 10 gen)'],
         '#F3E5F5', title_size=7, item_size=5.8)

# Arrows from Row 2 to Row 3
draw_arrow(2.6, 7.7, 3.0, 7.3)
draw_arrow(7.3, 7.7, 7.0, 7.3)

# ═══════════════════════════════════════
# ROW 3.5: COMMON MECHANISMS (y=3.6 to 4.5)
# ═══════════════════════════════════════

draw_box(0.5, 3.5, 4.2, 1.05, 'Constraint Handling',
         ["Deb's feasibility-first rules",
          'CV(x) = balance + ramp + reserve violations',
          'Repair operator for power balance'],
         '#E8F6F3', title_size=7.5, item_size=6)

draw_box(5.2, 3.5, 4.3, 1.05, 'Evaluation Framework',
         ['Deterministic mode: original RE forecast',
          'Robust mode: derated RE (Γ worst hours)',
          'Population size 80 × 200 generations ≈ 16,000 NFE'],
         '#E8F6F3', title_size=7.5, item_size=6)

# Arrows from algorithms to mechanisms
draw_arrow(2.6, 5.05, 2.6, 4.55)
draw_arrow(7.3, 5.05, 7.3, 4.55)

# ═══════════════════════════════════════
# ROW 4: OUTPUTS (y=1.6 to 3.0)
# ═══════════════════════════════════════

draw_box(0.3, 1.6, 3.2, 1.5, 'Pareto-Optimal Outputs',
         ['24-hour dispatch schedules',
          'Cost–emission–curtailment fronts',
          'Robust vs. deterministic comparison',
          'Best-compromise solutions'],
         c_output, item_size=6.2)

draw_box(3.8, 1.6, 2.9, 1.5, 'Performance Metrics',
         ['Hypervolume (MC, 80K samples)',
          'Spacing',
          'Computation time',
          'Final archive size'],
         c_output, item_size=6.2)

draw_box(7.0, 1.6, 2.7, 1.5, 'Statistical Validation',
         ['720 total runs (30 × 24)',
          'Mann-Whitney U test (α=0.05)',
          'Rank-biserial effect size',
          'Holm correction'],
         c_stat, item_size=6.2)

# Arrows from mechanisms to outputs
draw_arrow(2.6, 3.5, 1.9, 3.1)
draw_arrow(5.0, 3.5, 5.25, 3.1)
draw_arrow(7.4, 3.5, 8.3, 3.1)

# ═══════════════════════════════════════
# ROW 5: BOTTOM SUMMARY (y=0.4 to 1.2)
# ═══════════════════════════════════════

summary_box = FancyBboxPatch((1.5, 0.3), 7.0, 0.9, boxstyle="round,pad=0.08",
                              facecolor='#D4E6F1', edgecolor='#1A5276', linewidth=1.5)
ax.add_patch(summary_box)
ax.text(5.0, 0.85, 'IEEE 30-Bus: 6 Generators × 24h = 144 Decision Variables  |  '
        'NREL 118-Bus: 54 Generators × 24h = 1,296 Decision Variables',
        ha='center', va='center', fontsize=6.5, fontweight='bold',
        color='#1A5276', fontfamily='serif')
ax.text(5.0, 0.55, '3 Renewable Scenarios × 2 Dispatch Modes × 4 Algorithms × 30 Seeds = 720 Optimization Runs',
        ha='center', va='center', fontsize=6.5, color='#1A5276', fontfamily='serif')

# Arrows from outputs to summary
draw_arrow(1.9, 1.6, 3.5, 1.2)
draw_arrow(5.25, 1.6, 5.0, 1.2)
draw_arrow(8.3, 1.6, 6.5, 1.2)

plt.tight_layout(pad=0.3)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures', 'fig_block_diagram.png')
plt.savefig(OUT, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Block diagram saved: {OUT}")
