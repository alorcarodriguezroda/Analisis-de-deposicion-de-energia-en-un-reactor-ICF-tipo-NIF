import itertools
import pandas as pd
import numpy as np
from scipy.special import sph_harm


def write(df, filename):
    """
    Guarda un DataFrame en formato HDF5 con una clave estándar.
    """
    df.to_hdf(filename, key="impactos", mode="w")

def compute_sph_coeffs(hist, theta_edges, phi_edges, Lmax):

    # Centros de los bins
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_centers   = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    # Meshgrid con el mismo orden que hist: (theta, phi)
    mt, mp = np.meshgrid(theta_centers, phi_centers, indexing='ij')
    # mt = theta, mp = phi

    # Elemento de área
    dphi = phi_edges[1:] - phi_edges[:-1]
    dtheta_cos = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])
    S = np.outer(dtheta_cos, dphi)   # (n_theta x n_phi)

    # Lista de pares (l,m)
    lm = [(l, m) for l in range(Lmax + 1) for m in range(-l, l + 1)]

    def func_coef(l, m):
        Ylm = sph_harm(m, l, mp, mt)
        alm = np.sum(hist * np.conjugate(Ylm) * S)
        wl = np.sum(hist * S)
        return alm / wl

    return {(l, m): func_coef(l, m) for (l, m) in lm}



def print_coeffs(coeffs):
    print("\n================ COEFICIENTES NORMALIZADOS a_{lm} =================\n")
    print(f"{'l':>3} {'m':>4} {'Re(a_lm)':>15} {'Im(a_lm)':>15} {'|a_lm|':>15} {'|a_lm|/|a_00|':>15}")
    print("-" * 75)
    
    a_00 = coeffs[(0,0)]
    norm = abs(a_00)

    for (l, m), val in sorted(coeffs.items()):
        if norm == 0:
            ratio = float('nan')   # Normalización no definida
        else:
            ratio = abs(val)/norm

        print(f"{l:3d} {m:4d} {val.real:15.6e} {val.imag:15.6e} {abs(val):15.6e} {ratio:15.6e}")

    print("\n===================================================================\n")


