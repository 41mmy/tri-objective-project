#!/usr/bin/env python3
"""Unified HADE-NS, NSGA-II, MOPSO and MOEA/D, used unchanged on both test systems."""
import time
from math import comb

import numpy as np

from moeld_systems import FEAS_TOL, GAMMA, SYSTEMS

ARCHIVE_POLICIES = {
    'v1': {'all_evaluated': False, 'cap_factor': 1},
    'v2': {'all_evaluated': True, 'cap_factor': 2},
}


class Problem:
    def __init__(self, system_key, penetration_pct, robust, archive_policy='v2'):
        self.S = SYSTEMS[system_key]
        self.archive_all = ARCHIVE_POLICIES[archive_policy]['all_evaluated']
        self.cap = ARCHIVE_POLICIES[archive_policy]['cap_factor'] * self.S.pop_size
        self.demand = self.S.demand()
        self.re_avail = self.S.re_avail(penetration_pct)
        self.robust = robust
        self.gamma = GAMMA if robust else 0
        self.re_eff = self.S.effective_re(self.re_avail, robust, self.gamma)
        self.nfe = 0

    def evaluate(self, X):
        X = np.atleast_2d(X)
        self.nfe += X.shape[0]
        return self.S.evaluate(X, self.demand, self.re_avail, self.robust, self.gamma)

    def repair(self, X):
        return self.S.repair(X, self.demand, self.re_eff)


def fast_nds(obj, cv=None):
    n = obj.shape[0]
    feasible = cv <= FEAS_TOL if cv is not None else np.ones(n, dtype=bool)
    feas_idx, infeas_idx = np.where(feasible)[0], np.where(~feasible)[0]
    fronts = []
    if len(feas_idx) > 1:
        f = obj[feas_idx]
        dom = np.all(f[:, None, :] <= f[None, :, :], axis=2) & np.any(f[:, None, :] < f[None, :, :], axis=2)
        np.fill_diagonal(dom, False)
        count = dom.sum(axis=0)
        assigned = np.zeros(len(feas_idx), dtype=bool)
        while not np.all(assigned):
            layer = np.where((count == 0) & ~assigned)[0]
            fronts.append(feas_idx[layer])
            assigned[layer] = True
            count = count - dom[layer].sum(axis=0)
    elif len(feas_idx) == 1:
        fronts.append(feas_idx)
    if len(infeas_idx):
        fronts.append(infeas_idx[np.argsort(cv[infeas_idx], kind='stable')])
    return fronts


def crowding(obj):
    n, m = obj.shape
    if n <= 2:
        return np.full(n, np.inf)
    d = np.zeros(n)
    for k in range(m):
        order = np.argsort(obj[:, k], kind='stable')
        d[order[0]] = d[order[-1]] = np.inf
        rng_k = obj[order[-1], k] - obj[order[0], k]
        if rng_k > 1e-12:
            d[order[1:-1]] += (obj[order[2:], k] - obj[order[:-2], k]) / rng_k
    return d


def unique_idx(obj, scale, decimals=9):
    _, idx = np.unique(np.round(obj / scale, decimals), axis=0, return_index=True)
    return np.sort(idx)


def archive_update(arch, X, F, cv, cap, scale):
    feas = cv <= FEAS_TOL
    if not np.any(feas):
        return arch
    AX = np.vstack([arch[0], X[feas]])
    AF = np.vstack([arch[1], F[feas]])
    AC = np.concatenate([arch[2], cv[feas]])
    u = unique_idx(AF, scale)
    AX, AF, AC = AX[u], AF[u], AC[u]
    nd = fast_nds(AF)[0]
    AX, AF, AC = AX[nd], AF[nd], AC[nd]
    if len(AF) > cap:
        keep = np.argsort(-crowding(AF), kind='stable')[:cap]
        AX, AF, AC = AX[keep], AF[keep], AC[keep]
    return AX, AF, AC


def empty_archive(dim):
    return np.empty((0, dim)), np.empty((0, 3)), np.empty(0)


def constrained_better(f1, c1, f2, c2):
    if c1 <= FEAS_TOL and c2 <= FEAS_TOL:
        return bool(np.all(f1 <= f2) and np.any(f1 < f2))
    if c1 <= FEAS_TOL:
        return True
    if c2 <= FEAS_TOL:
        return False
    return c1 < c2


def select_survivors(obj, cv, k):
    chosen = []
    for front in fast_nds(obj, cv):
        if len(chosen) + len(front) <= k:
            chosen.extend(front.tolist())
            continue
        need = k - len(chosen)
        if cv[front[0]] > FEAS_TOL:
            chosen.extend(front[:need].tolist())
        else:
            chosen.extend(front[np.argsort(-crowding(obj[front]), kind='stable')[:need]].tolist())
        break
    return np.array(chosen, dtype=int)


def init_population(prob, n, rng):
    S = prob.S
    X = rng.uniform(S.LOWER, S.UPPER, (n, S.DIM))
    for i in range(n // 2):
        P = np.empty((S.N_GEN, S.HOURS))
        for t in range(S.HOURS):
            need = np.clip(prob.demand[t] - prob.re_eff[t], S.TOTAL_PMIN, S.TOTAL_PMAX)
            share = rng.dirichlet(np.ones(S.N_GEN))
            P[:, t] = S.P_MIN + share * (need - S.TOTAL_PMIN) + rng.randn(S.N_GEN) * (S.P_MAX - S.P_MIN) * 0.05
        X[i] = np.clip(P, S.P_MIN[:, None], S.P_MAX[:, None]).ravel()
    return prob.repair(X)


def poly_mutation(x, rng, pm, lower, upper, eta=20):
    x = x.copy()
    mask = rng.rand(x.shape[-1]) < pm
    if np.any(mask):
        u = rng.rand(mask.sum())
        step = np.where(u < 0.5, (2 * u) ** (1 / (eta + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta + 1)))
        x[mask] += step * (upper[mask] - lower[mask])
    return np.clip(x, lower, upper)


def _result(arch, t0, prob, n_gen):
    return {'X': arch[0], 'F': arch[1], 'cv': arch[2], 'time': time.time() - t0, 'nfe': prob.nfe, 'n_gen': n_gen}


def hade_ns(prob, seed, c=0.1, ls_every=15, ls_frac=0.05, ls_sigma=0.005):
    S, rng, t0 = prob.S, np.random.RandomState(seed), time.time()
    N, D, span = S.pop_size, S.DIM, S.UPPER - S.LOWER
    pop = init_population(prob, N, rng)
    obj, cv = prob.evaluate(pop)
    scale = np.maximum(np.abs(obj).max(axis=0), 1e-9)
    arch = archive_update(empty_archive(D), pop, obj, cv, prob.cap, scale)
    mu_f, mu_cr = 0.5, 0.5

    for gen in range(S.n_gen):
        F = np.clip(mu_f + 0.1 * rng.standard_cauchy(N), 0.1, 1.0)
        CR = np.clip(rng.normal(mu_cr, 0.1, N), 0.1, 0.9)
        trial = np.empty_like(pop)
        for i in range(N):
            r = rng.choice(N - 1, 3, replace=False)
            r[r >= i] += 1
            mutant = pop[r[0]] + F[i] * (pop[r[1]] - pop[r[2]])
            mask = rng.rand(D) < CR[i]
            mask[rng.randint(D)] = True
            trial[i] = np.clip(np.where(mask, mutant, pop[i]), S.LOWER, S.UPPER)
        trial = prob.repair(trial)
        t_obj, t_cv = prob.evaluate(trial)

        m_pop, m_obj, m_cv = np.vstack([pop, trial]), np.vstack([obj, t_obj]), np.concatenate([cv, t_cv])
        sel = select_survivors(m_obj, m_cv, N)
        won = sel[sel >= N] - N                              # trials that survived selection
        if len(won):
            mu_f = (1 - c) * mu_f + c * np.sum(F[won] ** 2) / np.sum(F[won])
            mu_cr = (1 - c) * mu_cr + c * np.mean(CR[won])
        pop, obj, cv = m_pop[sel], m_obj[sel], m_cv[sel]
        if prob.archive_all:
            arch = archive_update(arch, trial, t_obj, t_cv, prob.cap, scale)
        else:
            arch = archive_update(arch, pop, obj, cv, prob.cap, scale)

        if ls_every and (gen + 1) % ls_every == 0:
            feas = np.where(cv <= FEAS_TOL)[0]
            if len(feas):
                ls_x, ls_f, ls_c = [], [], []
                for idx in rng.choice(feas, min(max(1, round(ls_frac * N)), len(feas)), replace=False):
                    cand = np.clip(pop[idx] + rng.randn(D) * span * ls_sigma, S.LOWER, S.UPPER)
                    cand = prob.repair(cand)
                    c_obj, c_cv = prob.evaluate(cand)
                    ls_x.append(cand[0]), ls_f.append(c_obj[0]), ls_c.append(c_cv[0])
                    if constrained_better(c_obj[0], c_cv[0], obj[idx], cv[idx]):
                        pop[idx], obj[idx], cv[idx] = cand[0], c_obj[0], c_cv[0]
                if prob.archive_all:
                    arch = archive_update(arch, np.array(ls_x), np.array(ls_f), np.array(ls_c), prob.cap, scale)
                else:
                    arch = archive_update(arch, pop, obj, cv, prob.cap, scale)
    return _result(arch, t0, prob, S.n_gen)


def _tournament(rank, crowd, cv, k, rng):
    n = len(rank)
    a, b = rng.randint(0, n, k), rng.randint(0, n, k)
    infeas = (cv[a] > FEAS_TOL) | (cv[b] > FEAS_TOL)
    a_wins = np.where(infeas, cv[a] < cv[b], (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b])))
    return np.where(a_wins, a, b)


def nsga2(prob, seed, eta_c=20, pc=0.9, p_var=0.5, eta_m=20):
    S, rng, t0 = prob.S, np.random.RandomState(seed), time.time()
    N, D = S.pop_size, S.DIM
    pm = 1.0 / D
    pop = init_population(prob, N, rng)
    obj, cv = prob.evaluate(pop)
    scale = np.maximum(np.abs(obj).max(axis=0), 1e-9)
    arch = archive_update(empty_archive(D), pop, obj, cv, prob.cap, scale)

    for _ in range(S.n_gen):
        rank, crowd = np.full(N, N), np.zeros(N)
        for r, front in enumerate(fast_nds(obj, cv)):
            rank[front] = r
            crowd[front] = crowding(obj[front])
        parents = _tournament(rank, crowd, cv, N + (N % 2), rng)
        off = np.empty((len(parents), D))
        for k in range(0, len(parents), 2):
            p1, p2 = pop[parents[k]], pop[parents[k + 1]]
            c1, c2 = p1.copy(), p2.copy()
            if rng.rand() < pc:
                u = rng.rand(D)
                beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
                do = rng.rand(D) < p_var
                c1 = np.where(do, 0.5 * ((1 + beta) * p1 + (1 - beta) * p2), p1)
                c2 = np.where(do, 0.5 * ((1 - beta) * p1 + (1 + beta) * p2), p2)
            off[k] = poly_mutation(np.clip(c1, S.LOWER, S.UPPER), rng, pm, S.LOWER, S.UPPER, eta_m)
            off[k + 1] = poly_mutation(np.clip(c2, S.LOWER, S.UPPER), rng, pm, S.LOWER, S.UPPER, eta_m)
        off = prob.repair(off[:N])
        o_obj, o_cv = prob.evaluate(off)

        m_pop, m_obj, m_cv = np.vstack([pop, off]), np.vstack([obj, o_obj]), np.concatenate([cv, o_cv])
        u = unique_idx(m_obj, scale)
        chosen = u[select_survivors(m_obj[u], m_cv[u], N)]
        if len(chosen) < N:
            dup = np.setdiff1d(np.arange(len(m_obj)), u)
            chosen = np.concatenate([chosen, dup[:N - len(chosen)]])
        pop, obj, cv = m_pop[chosen], m_obj[chosen], m_cv[chosen]
        if prob.archive_all:
            arch = archive_update(arch, off, o_obj, o_cv, prob.cap, scale)
        else:
            arch = archive_update(arch, pop, obj, cv, prob.cap, scale)
    return _result(arch, t0, prob, S.n_gen)


def mopso(prob, seed, c1=2.0, c2=2.0, w_max=0.9, w_min=0.4, vmax_frac=0.2,
          mut_rate=0.1, mut_share=0.05, mut_sigma=0.1):
    S, rng, t0 = prob.S, np.random.RandomState(seed), time.time()
    N, D, span, G = S.pop_size, S.DIM, S.UPPER - S.LOWER, S.n_gen
    vmax = vmax_frac * span
    pos = init_population(prob, N, rng)
    vel = rng.uniform(-0.1 * vmax, 0.1 * vmax, (N, D))
    obj, cv = prob.evaluate(pos)
    scale = np.maximum(np.abs(obj).max(axis=0), 1e-9)
    pb, pb_obj, pb_cv = pos.copy(), obj.copy(), cv.copy()
    arch = archive_update(empty_archive(D), pos, obj, cv, prob.cap, scale)
    n_mut = max(1, int(D * mut_share))

    for gen in range(G):
        w = w_max - (w_max - w_min) * gen / G
        if len(arch[1]):
            cd = crowding(arch[1])
            cd = np.where(np.isinf(cd), 1e6, cd)
            prob_leader = cd / cd.sum() if cd.sum() > 0 else np.full(len(cd), 1 / len(cd))
        for i in range(N):
            leader = arch[0][rng.choice(len(arch[1]), p=prob_leader)] if len(arch[1]) else pb[np.argmin(pb_cv)]
            r1, r2 = rng.rand(D), rng.rand(D)
            vel[i] = np.clip(w * vel[i] + c1 * r1 * (pb[i] - pos[i]) + c2 * r2 * (leader - pos[i]), -vmax, vmax)
            pos[i] = pos[i] + vel[i]
            if rng.rand() < mut_rate * (1 - gen / G):
                dims = rng.choice(D, n_mut, replace=False)
                pos[i, dims] += rng.randn(n_mut) * span[dims] * mut_sigma
            pos[i] = np.clip(pos[i], S.LOWER, S.UPPER)
        pos = prob.repair(pos)
        obj, cv = prob.evaluate(pos)
        for i in range(N):
            if constrained_better(obj[i], cv[i], pb_obj[i], pb_cv[i]):
                pb[i], pb_obj[i], pb_cv[i] = pos[i], obj[i], cv[i]
            elif (cv[i] <= FEAS_TOL and pb_cv[i] <= FEAS_TOL
                  and not constrained_better(pb_obj[i], pb_cv[i], obj[i], cv[i]) and rng.rand() < 0.5):
                pb[i], pb_obj[i], pb_cv[i] = pos[i], obj[i], cv[i]
        arch = archive_update(arch, pos, obj, cv, prob.cap, scale)
    return _result(arch, t0, prob, G)


def lattice_weights(n_target, m=3):
    H = min(range(1, 200), key=lambda h: (abs(comb(h + m - 1, m - 1) - n_target), -h))
    W = []

    def rec(prefix, left, k):
        if k == 1:
            W.append(prefix + [left])
            return
        for v in range(left, -1, -1):
            rec(prefix + [v], left - v, k - 1)
    rec([], H, m)
    return np.array(W, dtype=float) / H


def moead(prob, seed, T=20, delta=0.9, nr=2, F=0.5, CR=1.0, eta_m=20):
    S, rng, t0 = prob.S, np.random.RandomState(seed), time.time()
    D = S.DIM
    W = lattice_weights(S.pop_size)
    n_w = len(W)
    n_gen = int(round(S.budget / n_w)) - 1                   # same evaluation budget as the others
    pm = 1.0 / D
    B = np.argsort(np.linalg.norm(W[:, None, :] - W[None, :, :], axis=2), axis=1, kind='stable')[:, :T]

    pop = init_population(prob, n_w, rng)
    obj, cv = prob.evaluate(pop)
    scale = np.maximum(np.abs(obj).max(axis=0), 1e-9)
    feas = cv <= FEAS_TOL
    z = obj[feas].min(axis=0) if np.any(feas) else obj.min(axis=0)
    arch = archive_update(empty_archive(D), pop, obj, cv, prob.cap, scale)

    def g_te(f, w, span):
        return np.max(np.maximum(w, 1e-6) * (f - z) / span)

    for _ in range(n_gen):
        feas = cv <= FEAS_TOL
        znad = obj[feas].max(axis=0) if np.any(feas) else obj.max(axis=0)
        span = np.maximum(znad - z, 1e-9 * np.maximum(np.abs(z), 1.0))
        kid_x, kid_f, kid_c = [], [], []
        for i in rng.permutation(n_w):
            pool = B[i] if rng.rand() < delta else np.arange(n_w)
            cand = pool[pool != i]
            r1, r2, r3 = rng.choice(cand, 3, replace=False)
            mask = rng.rand(D) < CR
            mask[rng.randint(D)] = True
            child = np.clip(np.where(mask, pop[r1] + F * (pop[r2] - pop[r3]), pop[i]), S.LOWER, S.UPPER)
            child = prob.repair(poly_mutation(child, rng, pm, S.LOWER, S.UPPER, eta_m))[0]
            c_obj, c_cv = prob.evaluate(child)
            c_obj, c_cv = c_obj[0], c_cv[0]
            kid_x.append(child), kid_f.append(c_obj), kid_c.append(c_cv)
            if c_cv <= FEAS_TOL:
                z = np.minimum(z, c_obj)
                span = np.maximum(znad - z, 1e-9 * np.maximum(np.abs(z), 1.0))
            replaced = 0
            for j in rng.permutation(pool):
                if c_cv > FEAS_TOL or cv[j] > FEAS_TOL:
                    better = c_cv < cv[j]
                else:
                    better = g_te(c_obj, W[j], span) < g_te(obj[j], W[j], span)
                if better:
                    pop[j], obj[j], cv[j] = child, c_obj, c_cv
                    replaced += 1
                    if replaced >= nr:
                        break
        if prob.archive_all:
            arch = archive_update(arch, np.array(kid_x), np.array(kid_f), np.array(kid_c), prob.cap, scale)
        else:
            arch = archive_update(arch, pop, obj, cv, prob.cap, scale)
    return _result(arch, t0, prob, n_gen)


ALGORITHMS = {'HADE-NS': hade_ns, 'NSGA-II': nsga2, 'MOPSO': mopso, 'MOEA/D': moead}


def run(system_key, algorithm, penetration_pct, robust, seed, archive_policy='v2'):
    prob = Problem(system_key, penetration_pct, robust, archive_policy)
    return ALGORITHMS[algorithm](prob, seed)
