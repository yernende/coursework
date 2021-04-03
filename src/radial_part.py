import numpy as np
from scipy.integrate import solve_ivp

from src.constants import *

def estimate_zero_derivative(l):
    return {
        0: 2.0,
        1: 1e-7,
        2: 1e-16,
        3: 1e-21,
        4: 1e-27,
        5: 1e-32,
        6: 1e-37,
        7: 1e-42,
        8: 1e-46,
        9: 1e-51,
        10: 1e-55,
        29: 1e-110,
        34: 1e-119,
        50: 1e-163,
        80: 1e-211,
        100: 1e-238
    }.get(l, 10 ** (-296.85135561489 + 288.50969096897 * np.exp(-l / 64.900741161138))) * np.random.randint(1, 100)

def compute_P(energy, l=0, atomic_number=1):
    """Compute the radial part of the HF wave function of an electron"""
    zero_derivative = estimate_zero_derivative(l)

    def get_potential(radius):
            return 2 * atomic_number / radius

    def ode_function(radius, y):
        return np.array([
            y[1],
            zero_derivative if radius == 0
                else (energy + l * (l + 1) / radius ** 2 - get_potential(radius)) * y[0]
        ])

    return solve_ivp(
        ode_function, [0, RADIAL_PART_MAX], [0.0, zero_derivative],
        first_step=RADIAL_PART_STEP, max_step=RADIAL_PART_STEP
    )
