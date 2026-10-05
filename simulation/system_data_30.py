"""IEEE 30-bus six-unit fleet: generator limits, cost and emission coefficients, ramp rates,
and the 24-hour load, wind and solar profiles."""
import numpy as np

N_GEN = 6

HOURS = 24

P_MIN = np.array([50.0, 20.0, 15.0, 10.0, 10.0, 12.0])

P_MAX = np.array([200.0, 80.0, 50.0, 35.0, 30.0, 40.0])

COST_A = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

COST_B = np.array([2.00, 1.75, 1.00, 3.25, 3.00, 3.00])

COST_C = np.array([0.00375, 0.01750, 0.06250, 0.00834, 0.02500, 0.02500])

EMIT_A = np.array([0.0169, 0.0102, 0.0050, 0.0213, 0.0060, 0.0120])

EMIT_B = np.array([0.000320, 0.000195, 0.000050, 0.000178, 0.000080, 0.000161])

EMIT_C = np.array([3.50e-6, 2.81e-6, 0.50e-6, 3.50e-6, 0.80e-6, 1.80e-6])

RAMP_RATE = np.array([40.0, 20.0, 15.0, 10.0, 10.0, 12.0])

BASE_DEMAND = 283.4

RESERVE_FACTOR = 0.10

LOAD_FACTORS = np.array([
    0.62, 0.60, 0.58, 0.56, 0.58, 0.62,
    0.68, 0.78, 0.88, 0.92, 0.94, 0.95,
    0.93, 0.90, 0.88, 0.87, 0.90, 0.95,
    0.98, 1.00, 0.97, 0.92, 0.82, 0.70
])


def generate_load_profile():
    return LOAD_FACTORS * BASE_DEMAND


def generate_solar_profile(capacity):
    solar = np.zeros(HOURS)
    for h in range(6, 19):
        solar[h] = np.sin(np.pi * (h - 6) / 12) * 0.85
    rng = np.random.RandomState(123)
    solar *= (1 + 0.03 * rng.randn(HOURS))
    solar = np.clip(solar, 0, 1.0)
    return solar * capacity


def generate_wind_profile(capacity):
    rng = np.random.RandomState(456)
    base_cf = np.array([
        0.40, 0.42, 0.45, 0.48, 0.45, 0.40,
        0.35, 0.30, 0.28, 0.25, 0.22, 0.20,
        0.22, 0.25, 0.28, 0.32, 0.35, 0.38,
        0.42, 0.45, 0.48, 0.50, 0.48, 0.45
    ])
    noise = np.zeros(HOURS)
    for h in range(HOURS):
        noise[h] = (0.7 * noise[h-1] if h > 0 else 0) + 0.08 * rng.randn()
    return np.clip(base_cf + noise, 0.05, 0.85) * capacity


def get_re_profiles(penetration_pct):
    total_re = penetration_pct / 100.0 * BASE_DEMAND
    wind_cap = 0.6 * total_re
    solar_cap = 0.4 * total_re
    return generate_wind_profile(wind_cap) + generate_solar_profile(solar_cap)
