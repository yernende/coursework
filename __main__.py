import os
from multiprocessing import Pool

import numpy as np
import scipy.linalg
from matplotlib import pyplot

from src.classes import Atom
from src.classes import Vector
from src.matrix import compute_matrix
from src.radial_part import compute_P
from src.constants import *

# Hydrogen ion
# 1.06 Angstrom == 2.003109692 bohr
# 124429 cm^-1 == 1.133880478 Ry

atoms = [Atom([0, 0, 0], 1), Atom([0, 0, 2.003109692], 1)]
energies = np.linspace(1.0, 1.5, 20)
vectors = [Vector([0, 0, t]) for t in np.linspace(0.5, 1.5, 11)]

def compute_for_fixed_energy(energy):
    results_for_fixed_energy = []

    for atom in atoms:
        atom.radial_part_derivative_to_itself_ratio_array = np.zeros(MAX_L)

        for l in range(MAX_L):
            P = compute_P(energy, l, atom.atomic_number)
            atom.radial_part_derivative_to_itself_ratio_array[l] = (
                P.y[1][int(atom.radius_muffintin / RADIAL_PART_STEP)]
                / P.y[0][int(atom.radius_muffintin / RADIAL_PART_STEP)]
            )

    matrix = compute_matrix(atoms, vectors, energy)
    determinant = abs(scipy.linalg.det(matrix))

    results_for_fixed_energy.append({
        'energy': energy,
        'determinant': determinant
    })

    print(results_for_fixed_energy[-1])
    return results_for_fixed_energy

if __name__ == '__main__':
    with Pool(os.cpu_count()) as pool:
        results = []

        for results_for_fixed_energy in pool.imap_unordered(
            compute_for_fixed_energy,
            energies,
            int(len(energies) / os.cpu_count())
        ):
            results += results_for_fixed_energy

        print('\nFinal result')
        print(min(results, key=lambda result: result['determinant']))

        X, Y = [], []

        for result in sorted(results, key=lambda result: result['energy']):
            X.append(result['energy'])
            Y.append(result['determinant'])

        X, Y = np.array(X), np.array(Y)

        pyplot.plot(X, Y)
        pyplot.show()
