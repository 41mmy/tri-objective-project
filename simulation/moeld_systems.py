#!/usr/bin/env python3
"""Test systems (IEEE 30-bus, NREL-118) with one shared evaluate() and ramp-aware repair()."""
import numpy as np

import system_data_118 as _m118
import system_data_30 as _m30

FEAS_TOL = 1e-2
UNCERTAINTY = 0.15  # robust mode: downside derating of the Gamma highest-renewable hours
GAMMA = 6


class System:
    def __init__(self, name, mod, pop_size, n_gen):
        self.name = name
        self._mod = mod
        self.N_GEN, self.HOURS = mod.N_GEN, mod.HOURS
        self.DIM = self.N_GEN * self.HOURS
        self.P_MIN, self.P_MAX = mod.P_MIN.astype(float), mod.P_MAX.astype(float)
        self.TOTAL_PMIN, self.TOTAL_PMAX = self.P_MIN.sum(), self.P_MAX.sum()
        self.LOWER, self.UPPER = np.repeat(self.P_MIN, self.HOURS), np.repeat(self.P_MAX, self.HOURS)
        self.COST = (mod.COST_A, mod.COST_B, mod.COST_C)
        self.EMIT = (mod.EMIT_A, mod.EMIT_B, mod.EMIT_C)
        self.RAMP = mod.RAMP_RATE.astype(float)
        self.RESERVE = mod.RESERVE_FACTOR
        self.pop_size, self.n_gen = pop_size, n_gen
        self.budget = pop_size * (n_gen + 1)

    def demand(self):
        return self._mod.generate_load_profile()

    def re_avail(self, penetration_pct):
        return self._mod.get_re_profiles(penetration_pct)

    @staticmethod
    def effective_re(re_avail, robust, gamma=GAMMA):
        re_eff = re_avail.copy()
        if robust and gamma > 0:
            unc = UNCERTAINTY * re_avail
            worst = np.argsort(-unc)[:gamma]
            re_eff[worst] -= unc[worst]
            re_eff = np.clip(re_eff, 0, None)
        return re_eff

    def evaluate(self, X, demand, re_avail, robust, gamma=GAMMA):
        X = np.atleast_2d(X)
        n, G, H = X.shape[0], self.N_GEN, self.HOURS
        P = np.clip(X.reshape(n, G, H), self.P_MIN[None, :, None], self.P_MAX[None, :, None])
        re_eff = self.effective_re(re_avail, robust, gamma)

        def quad(coef):
            a, b, c = (np.asarray(v).reshape(1, G, 1) for v in coef)
            return np.sum(a + b * P + c * P ** 2, axis=(1, 2))

        thermal = P.sum(axis=1)
        accepted = np.clip(demand[None, :] - thermal, 0, re_eff[None, :])
        curtailment = np.sum(np.maximum(re_eff[None, :] - accepted, 0), axis=1)
        cv_balance = np.sum(np.abs(thermal + accepted - demand[None, :]), axis=1)
        cv_excess = np.sum(np.maximum(thermal - demand[None, :], 0), axis=1)
        cv_ramp = np.sum(np.maximum(np.abs(np.diff(P, axis=2)) - self.RAMP[None, :, None], 0), axis=(1, 2))
        reserve = np.sum(self.P_MAX[None, :, None] - P, axis=1)
        cv_reserve = np.sum(np.maximum(self.RESERVE * demand[None, :] - reserve, 0), axis=1)
        obj = np.column_stack([quad(self.COST), quad(self.EMIT), curtailment])
        return obj, cv_balance + cv_excess + cv_ramp + cv_reserve

    def repair(self, X, demand, re_eff):
        X = np.atleast_2d(X)
        n, G, H = X.shape[0], self.N_GEN, self.HOURS
        P = X.reshape(n, G, H).copy()
        pmin = np.broadcast_to(self.P_MIN[None, :], (n, G))
        pmax = np.broadcast_to(self.P_MAX[None, :], (n, G))
        ramp = self.RAMP[None, :]
        for t in range(H):
            if t == 0:
                lo, hi = pmin, pmax
            else:
                lo = np.maximum(pmin, P[:, :, t - 1] - ramp)
                hi = np.minimum(pmax, P[:, :, t - 1] + ramp)
            P[:, :, t] = np.clip(P[:, :, t], lo, hi)
            target_min = max(demand[t] - re_eff[t], self.TOTAL_PMIN)
            target_max = demand[t]
            total = P[:, :, t].sum(axis=1)
            low = total < target_min - 0.1
            if np.any(low):
                room = hi[low] - P[low, :, t]
                scale = np.minimum((target_min - total[low])[:, None] / np.maximum(room.sum(1, keepdims=True), 1e-6), 1.0)
                P[low, :, t] += room * scale
            high = total > target_max + 0.1
            if np.any(high):
                room = P[high, :, t] - lo[high]
                scale = np.minimum((total[high] - target_max)[:, None] / np.maximum(room.sum(1, keepdims=True), 1e-6), 1.0)
                P[high, :, t] -= room * scale
            P[:, :, t] = np.clip(P[:, :, t], lo, hi)
        return P.reshape(n, self.DIM)


SYSTEMS = {
    '30': System('IEEE 30-bus', _m30, pop_size=80, n_gen=200),
    '118': System('NREL-118', _m118, pop_size=100, n_gen=250),
}
