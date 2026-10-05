#!/usr/bin/env python3
"""Generate HADE-NS Algorithm Flowchart for IEEE paper."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(7.5, 11.5), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 16)
ax.axis('off')

# Colors
c_start = '#2C3E50'     # dark - start/end
c_process = '#D6EAF8'   # blue - process
c_decision = '#FDEBD0'  # orange - decision
c_io = '#D5F5E3'        # green - input/output
c_sub = '#E8DAEF'        # purple - subroutine
c_loop = '#FCE4EC'       # pink - loop header
border = '#2C3E50'
text_c = '#1B2631'

def draw_rounded(x, y, w, h, text, color, fontsize=7, text_color='white', lw=1.2, lines=None):
    """Rounded rectangle (start/end terminals)."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.12",
                          facecolor=color, edgecolor=border, linewidth=lw)
    ax.add_patch(box)
    if lines:
        for i, line in enumerate(lines):
            offset = (len(lines) - 1) / 2 - i
            ax.text(x, y + offset * 0.18, line, ha='center', va='center',
                    fontsize=fontsize, color=text_color, fontfamily='serif', fontweight='bold')
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, color=text_color, fontfamily='serif', fontweight='bold')

def draw_rect(x, y, w, h, lines, color, fontsize=6.5, bold_first=True):
    """Rectangle process box with multiple lines."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.06",
                          facecolor=color, edgecolor=border, linewidth=1.1)
    ax.add_patch(box)
    n = len(lines)
    for i, line in enumerate(lines):
        offset = (n - 1) / 2 - i
        fw = 'bold' if (i == 0 and bold_first) else 'normal'
        ax.text(x, y + offset * 0.2, line, ha='center', va='center',
                fontsize=fontsize, color=text_c, fontfamily='serif', fontweight=fw)

def draw_diamond(x, y, w, h, lines, color, fontsize=6.5):
    """Diamond decision box."""
    pts = [(x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)]
    diamond = plt.Polygon(pts, facecolor=color, edgecolor=border, linewidth=1.2)
    ax.add_patch(diamond)
    n = len(lines)
    for i, line in enumerate(lines):
        offset = (n - 1) / 2 - i
        ax.text(x, y + offset * 0.18, line, ha='center', va='center',
                fontsize=fontsize, color=text_c, fontfamily='serif', fontweight='bold')

def draw_parallelogram(x, y, w, h, lines, color, fontsize=6.5):
    """Parallelogram for I/O."""
    skew = 0.3
    pts = [(x - w/2 + skew, y + h/2), (x + w/2 + skew, y + h/2),
           (x + w/2 - skew, y - h/2), (x - w/2 - skew, y - h/2)]
    para = plt.Polygon(pts, facecolor=color, edgecolor=border, linewidth=1.1)
    ax.add_patch(para)
    n = len(lines)
    for i, line in enumerate(lines):
        offset = (n - 1) / 2 - i
        ax.text(x, y + offset * 0.2, line, ha='center', va='center',
                fontsize=fontsize, color=text_c, fontfamily='serif')

def arrow(x1, y1, x2, y2, color='#566573', lw=1.3):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

def arrow_label(x1, y1, x2, y2, label, side='right', color='#566573'):
    arrow(x1, y1, x2, y2, color)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx = 0.15 if side == 'right' else -0.15
    ax.text(mx + dx, my, label, fontsize=6, color='#E74C3C', fontfamily='serif',
            fontweight='bold', ha='left' if side == 'right' else 'right', va='center')

# Center x
cx = 5.0

# ── Title ──
ax.text(cx, 15.7, 'HADE-NS Algorithm Flowchart', ha='center', va='top',
        fontsize=12, fontweight='bold', color=text_c, fontfamily='serif')

# ═══════════════ STEP 1: START ═══════════════
y = 15.2
draw_rounded(cx, y, 3.0, 0.45, 'START', c_start, fontsize=9)

# ═══════════════ STEP 2: INPUT ═══════════════
y_input = 14.45
draw_parallelogram(cx, y_input, 4.5, 0.55, [
    'Input: System data, demand profile, RE profiles,',
    'penetration level, dispatch mode (Det./Rob.)'
], c_io, fontsize=6)
arrow(cx, 15.2 - 0.225, cx, y_input + 0.275)

# ═══════════════ STEP 3: INITIALIZE ═══════════════
y_init = 13.55
draw_rect(cx, y_init, 4.8, 0.65, [
    'Initialize Population (N = 80)',
    '50% random uniform within [Pᵢᵐⁱⁿ, Pᵢᵐᵃˣ]',
    '50% feasibility-guided (demand-aware Dirichlet)'
], c_process, fontsize=6.2)
arrow(cx, y_input - 0.275, cx, y_init + 0.325)

# ═══════════════ STEP 4: EVALUATE ═══════════════
y_eval = 12.65
draw_rect(cx, y_eval, 4.8, 0.55, [
    'Evaluate Objectives & Constraints',
    'f₁(cost), f₂(emissions), f₃(curtailment), CV(x)'
], c_process, fontsize=6.2)
arrow(cx, y_init - 0.325, cx, y_eval + 0.275)

# ═══════════════ STEP 4b: ROBUST CHECK ═══════════════
y_robust = 11.85
draw_diamond(cx, y_robust, 2.8, 0.65, ['Robust mode?'], c_decision, fontsize=7)
arrow(cx, y_eval - 0.275, cx, y_robust + 0.325)

# Yes branch - right
y_derate = 11.85
draw_rect(8.0, y_derate, 2.8, 0.55, [
    'Apply Bertsimas–Sim derating',
    'Select Γ=6 worst hours,',
    'reduce RE by 15%'
], c_sub, fontsize=5.8, bold_first=True)
arrow_label(cx + 1.4, y_robust, 8.0 - 1.4, y_derate, 'Yes', side='right')

# No branch continues down
# ═══════════════ LOOP HEADER ═══════════════
y_loop = 11.0
draw_rect(cx, y_loop, 4.8, 0.45, [
    'FOR gen = 1 TO 200 generations'
], c_loop, fontsize=7, bold_first=True)
arrow_label(cx, y_robust - 0.325, cx, y_loop + 0.225, 'No', side='right')

# Arrow from robust box back down
ax.annotate('', xy=(8.0, y_loop + 0.225), xytext=(8.0, y_derate - 0.275),
            arrowprops=dict(arrowstyle='->', color='#566573', lw=1.3))
ax.plot([cx + 2.4, 8.0], [y_loop + 0.225, y_loop + 0.225], color='#566573', lw=1.3)

# ═══════════════ STEP 5: MUTATION ═══════════════
y_mut = 10.25
draw_rect(cx, y_mut, 4.8, 0.55, [
    'DE/rand/1/bin Mutation & Crossover',
    'Adaptive Fᵢ ~ Cauchy(F_mean, 0.1),  CRᵢ ~ N(CR_mean, 0.1)',
    'Trial vector: uᵢ = binomial(xᵢ, vᵢ)'
], c_process, fontsize=6)
arrow(cx, y_loop - 0.225, cx, y_mut + 0.275)

# ═══════════════ STEP 6: EVALUATE TRIAL ═══════════════
y_teval = 9.45
draw_rect(cx, y_teval, 4.8, 0.45, [
    'Evaluate Trial Population',
    'Compute f₁, f₂, f₃, CV for all trial vectors'
], c_process, fontsize=6.2)
arrow(cx, y_mut - 0.275, cx, y_teval + 0.225)

# ═══════════════ STEP 7: MERGE & SORT ═══════════════
y_merge = 8.6
draw_rect(cx, y_merge, 4.8, 0.6, [
    'Merge & Environmental Selection',
    'Combine parent + trial (2N); fast nondominated sorting;',
    'crowding distance; Deb feasibility-first; select top N'
], c_process, fontsize=6)
arrow(cx, y_teval - 0.225, cx, y_merge + 0.3)

# ═══════════════ STEP 8: PARAMETER ADAPTATION ═══════════════
y_adapt = 7.75
draw_rect(cx, y_adapt, 4.8, 0.55, [
    'JADE Parameter Adaptation',
    'Update F_mean (Lehmer mean of successes)',
    'Update CR_mean (arithmetic mean of successes)'
], c_process, fontsize=6.2)
arrow(cx, y_merge - 0.3, cx, y_adapt + 0.275)

# ═══════════════ STEP 9: LOCAL SEARCH CHECK ═══════════════
y_ls = 6.95
draw_diamond(cx, y_ls, 3.2, 0.6, ['gen mod 15 = 0?'], c_decision, fontsize=6.5)
arrow(cx, y_adapt - 0.275, cx, y_ls + 0.3)

# Yes branch right
y_lsbox = 6.95
draw_rect(8.2, y_lsbox, 2.6, 0.7, [
    'Local Search',
    'n_ls = max(1, min(5, |feas|/10))',
    'Perturb ±0.5% of range',
    'Accept if dominates parent'
], c_sub, fontsize=5.5, bold_first=True)
arrow_label(cx + 1.6, y_ls, 8.2 - 1.3, y_lsbox, 'Yes', side='right')

# ═══════════════ STEP 10: TERMINATION CHECK ═══════════════
y_term = 6.05
draw_diamond(cx, y_term, 3.2, 0.6, ['gen < 200?'], c_decision, fontsize=7)
arrow_label(cx, y_ls - 0.3, cx, y_term + 0.3, 'No', side='right')

# Arrow from local search down to termination
ax.annotate('', xy=(8.2, y_term + 0.3), xytext=(8.2, y_lsbox - 0.35),
            arrowprops=dict(arrowstyle='->', color='#566573', lw=1.3))
ax.plot([cx + 1.6, 8.2], [y_term + 0.3, y_term + 0.3], color='#566573', lw=1.3)

# Yes: loop back — draw left side return
arrow_label(cx - 1.6, y_term, 1.2, y_term, 'Yes', side='left')
ax.plot([1.2, 1.2], [y_term, y_loop], color='#566573', lw=1.3)
ax.annotate('', xy=(cx - 2.4, y_loop), xytext=(1.2, y_loop),
            arrowprops=dict(arrowstyle='->', color='#566573', lw=1.3))

# No: continue down
# ═══════════════ STEP 11: ARCHIVE EXTRACTION ═══════════════
y_arch = 5.15
draw_rect(cx, y_arch, 4.8, 0.55, [
    'Extract Pareto Archive',
    'Identify nondominated feasible solutions',
    'Truncate to population size via crowding distance'
], c_process, fontsize=6.2)
arrow_label(cx, y_term - 0.3, cx, y_arch + 0.275, 'No', side='right')

# ═══════════════ STEP 12: REPEAT FOR BASELINES ═══════════════
y_base = 4.3
draw_rect(cx, y_base, 4.8, 0.55, [
    'Repeat for Baseline Algorithms',
    'NSGA-II (SBX + polynomial mutation)',
    'MOPSO (velocity + grid archive), MOEA/D (Tchebycheff)'
], c_sub, fontsize=6.2)
arrow(cx, y_arch - 0.275, cx, y_base + 0.275)

# ═══════════════ STEP 13: STATISTICAL COMPARISON ═══════════════
y_stat = 3.4
draw_rect(cx, y_stat, 4.8, 0.65, [
    'Performance Evaluation & Statistical Testing',
    'Hypervolume (MC, 80K samples), Spacing, Time',
    '30 independent seeds × 24 configurations = 720 runs',
    'Mann-Whitney U test (α = 0.05), Holm correction'
], c_process, fontsize=6)
arrow(cx, y_base - 0.275, cx, y_stat + 0.325)

# ═══════════════ STEP 14: OUTPUT ═══════════════
y_out = 2.4
draw_parallelogram(cx, y_out, 4.8, 0.65, [
    'Output: Pareto-optimal dispatch schedules,',
    'cost–emission–curtailment trade-off fronts,',
    'robust vs. deterministic comparison'
], c_io, fontsize=6.2)
arrow(cx, y_stat - 0.325, cx, y_out + 0.325)

# ═══════════════ END ═══════════════
y_end = 1.55
draw_rounded(cx, y_end, 3.0, 0.45, 'END', c_start, fontsize=9)
arrow(cx, y_out - 0.325, cx, y_end + 0.225)

# ═══════════════ SIDE ANNOTATIONS ═══════════════
# Loop bracket
ax.plot([0.7, 0.7], [y_loop + 0.225, y_term - 0.3], color='#E74C3C', lw=2, linestyle='--', alpha=0.5)
ax.text(0.35, (y_loop + y_term) / 2, 'Main\nLoop', ha='center', va='center',
        fontsize=6.5, color='#E74C3C', fontfamily='serif', fontweight='bold',
        rotation=90)

plt.tight_layout(pad=0.3)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures', 'fig_flowchart.png')
plt.savefig(OUT, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Flowchart saved: {OUT}")
