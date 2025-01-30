"""Functions for evaluating integrals"""

import torch
import numpy as np
from typing import List, Union

from orbitals import PrimitiveGaussian, ContractedGaussian
import chemical

import pdb


def primitive_gaussian_overlap(
    pg1: PrimitiveGaussian, pg2: PrimitiveGaussian
) -> torch.Tensor:
    """Compute the overlap between two primitive gaussians.

    References:
        Szabo and Ostlund, pg. 411
        Helgaker, Jorgensen, Olsen, pg. 341

    Args:
        pg1: first primitive gaussian
        pg2: second primitive gaussian
    """
    alpha = pg1.alpha
    beta = pg2.alpha

    p = alpha + beta
    mu = alpha * beta / p

    # convert from Angstroms to Bohr here...
    Ra = pg1.center * chemical.angstrom2bohr
    Rb = pg2.center * chemical.angstrom2bohr

    squared_distance = torch.sum((Ra - Rb) ** 2)

    K = torch.exp(-mu * squared_distance)

    overlap = ((np.pi / p) ** (3 / 2)) * K

    # normalize
    overlap *= pg1.prefactor * pg2.prefactor

    return overlap


def contracted_gaussian_overlap(cg1, cg2):
    """Compute single overlap s between two contracted gaussians."""
    assert len(cg1) == len(
        cg2
    ), "Contracted gaussians must have the same number of primitives"
    s = 0

    NG = len(cg1)
    d1 = cg1.coefficients
    d2 = cg2.coefficients

    for i in range(NG):
        for j in range(NG):
            prim_1 = cg1.primitives[i]
            prim_2 = cg2.primitives[j]

            s += d1[i] * d2[j] * primitive_gaussian_overlap(prim_1, prim_2)

    return s


def contracted_gaussian_overlap_matrix(
    cg_orbitals: List[ContractedGaussian],
) -> torch.Tensor:
    """Compute the overlap matrix for a set of contracted gaussians.

    Args:
        cg_orbitals: list of contracted gaussians
    """
    L = len(cg_orbitals)

    S = torch.zeros((L, L))

    # super dumb implementation, just to get things going
    for i in range(L):
        for j in range(i, L):

            S[i, j] = contracted_gaussian_overlap(cg_orbitals[i], cg_orbitals[j])
            S[j, i] = S[i, j]

    return S


all_contracted_gaussians = lambda x: all([isinstance(o, ContractedGaussian) for o in x])


def compute_overlap_matrix(basis_set: List):
    """Get the overlap matrix for a set of contracted gaussians.

    Args:
        basis_set: list of orbital objects
    """
    if all_contracted_gaussians(basis_set):  # contracted gaussians
        return contracted_gaussian_overlap_matrix(basis_set)

    else:
        raise NotImplementedError()


def primitive_gaussian_kinetic(pg1: PrimitiveGaussian, pg2: PrimitiveGaussian):
    """Compute (A|ke|B) between two primitive gaussians.

    Args:
        pg1: first primitive gaussian
        pg2: second primitive gaussian
    """
    alpha = pg1.alpha
    beta = pg2.alpha

    p = alpha + beta
    mu = alpha * beta / p

    # convert from Angstroms to Bohr here...
    Ra = pg1.center * chemical.angstrom2bohr
    Rb = pg2.center * chemical.angstrom2bohr

    squared_distance = torch.sum((Ra - Rb) ** 2)

    term1 = mu * (3 - 2 * mu * squared_distance)
    term2 = (np.pi / p) ** (3 / 2)
    term3 = torch.exp(-mu * squared_distance)

    ke = term1 * term2 * term3

    # normalize
    ke *= pg1.prefactor * pg2.prefactor

    return ke


def contracted_gaussian_kinetic(cg1: ContractedGaussian, cg2: ContractedGaussian):
    """Compute (A|ke|B) between two contracted gaussians."""

    T = 0
    NG = len(cg1)
    d1 = cg1.coefficients
    d2 = cg2.coefficients

    for i in range(NG):
        for j in range(NG):
            prim_1 = cg1.primitives[i]
            prim_2 = cg2.primitives[j]

            T += d1[i] * d2[j] * primitive_gaussian_kinetic(prim_1, prim_2)

    return T


def contracted_gaussian_T_matrix(cg_orbitals: List[ContractedGaussian]):
    """Compute the kinetic energy matrix for a set of contracted gaussians.

    Args:
        cg_orbitals: list of contracted gaussians
    """
    L = len(cg_orbitals)

    T = torch.zeros((L, L))

    for i in range(L):
        for j in range(i, L):

            T[i, j] = contracted_gaussian_kinetic(cg_orbitals[i], cg_orbitals[j])
            T[j, i] = T[i, j]

    return T


def Boys(t):
    # The boys function function from Szabo and Ostlund, pg. 415
    if t < 1e-10:
        # terrifying -- it gets the right answer, but how can we possibly get gradient through this?
        return 1.0

    A = 0.5 * ((np.pi / t) ** 0.5)
    B = torch.erf(t**0.5)
    return A * B


def primitive_gaussian_nuclear_attraction(
    pg1: PrimitiveGaussian, pg2: PrimitiveGaussian, nucleus: torch.Tensor
):
    """Compute (A|Vnuc|B) between two primitive gaussians.

    Args:
        pg1: first primitive gaussian
        pg2: second primitive gaussian
        nucleus: torch tensor of nucleus position
    """
    alpha = pg1.alpha
    beta = pg2.alpha

    p = alpha + beta
    mu = alpha * beta / p

    # convert from Angstroms to Bohr here...
    Ra = pg1.center * chemical.angstrom2bohr
    Rb = pg2.center * chemical.angstrom2bohr
    Rp = (alpha * Ra + beta * Rb) / (p)
    Rc = nucleus * chemical.angstrom2bohr

    squared_distance_ab = torch.sum((Ra - Rb) ** 2)
    squared_distance_cp = torch.sum((Rp - Rc) ** 2)

    term1 = -2 * np.pi / p
    term2 = torch.exp(-mu * squared_distance_ab)
    term3 = Boys(p * squared_distance_cp)

    V = term1 * term2 * term3

    # normalize
    V *= pg1.prefactor * pg2.prefactor

    return V


def contracted_gaussian_nuclear_attraction(
    cg1: ContractedGaussian, cg2: ContractedGaussian, nucleus: torch.Tensor
):
    """Compute (A|Vnuc|B) between two contracted gaussians."""
    V = 0
    NG = len(cg1)
    d1 = cg1.coefficients
    d2 = cg2.coefficients

    for i in range(NG):
        for j in range(NG):
            prim_1 = cg1.primitives[i]
            prim_2 = cg2.primitives[j]

            V += (
                d1[i]
                * d2[j]
                * primitive_gaussian_nuclear_attraction(prim_1, prim_2, nucleus)
            )

    return V


def contracted_gaussian_V_matrix(
    cg_orbitals: List[ContractedGaussian], nucleus: torch.Tensor
):
    """Compute the nuclear attraction matrix for a set of contracted gaussians.

    Args:
        cg_orbitals: list of contracted gaussians
        nucleus: torch tensor of nucleus position
    """
    L = len(cg_orbitals)

    V = torch.zeros((L, L))

    for i in range(L):
        for j in range(i, L):

            V[i, j] = contracted_gaussian_nuclear_attraction(
                cg_orbitals[i], cg_orbitals[j], nucleus
            )
            V[j, i] = V[i, j]

    return V


def contracted_gaussian_core_hamiltonian_matrix(cg_orbitals: List[ContractedGaussian]):
    """ """
    T = contracted_gaussian_T_matrix(cg_orbitals)
    V = contracted_gaussian_V_matrix(cg_orbitals)
    H = T + V
    return H


def compute_core_hamiltonian_matrix(basis_set: List):
    """Compute the core Hamiltonian matrix for a set of contracted gaussians.

    Args:
        basis_set: list of orbital objects
    """
    if all_contracted_gaussians(basis_set):
        return contracted_gaussian_core_hamiltonian_matrix(basis_set)


def unnormalized_primitive_gaussian_2e_integral(
    a: float, b: float, c: float, d: float, Ra: float, Rb: float, Rc: float, Rd: float
):
    """General integral (AB|CD) on four primitive gaussians.

    Args:
        a/b/c/d: exponents
        Ra/Rb/Rc/Rd: centers of the gaussians

    """
    d_ab_squared = torch.sum((Ra - Rb) ** 2)
    d_cd_squared = torch.sum((Rc - Rd) ** 2)

    Rp = (a * Ra + b * Rb) / (a + b)
    Rq = (c * Rc + d * Rd) / (c + d)
    d_pq_squared = torch.sum((Rp - Rq) ** 2)

    term1 = 2 * (np.pi ** (5 / 2)) / ((a + b) * (c + d) * torch.sqrt(a + b + c + d))
    term2 = torch.exp(
        (-a * b / (a + b)) * d_ab_squared - (c * d / (c + d)) * d_cd_squared
    )
    term3 = Boys((a + b) * (c + d) / (a + b + c + d) * d_pq_squared)

    return term1 * term2 * term3


def contracted_gaussian_2e_integral(cg1, cg2, cg3, cg4):
    """
    Computes two electron integral between four contracted gaussians.

    Args:
        cg1/2/3/4: contracted gaussians

    Returns:

    """
    assert (
        len(cg1) == len(cg2) == len(cg3) == len(cg4)
    ), "Contracted gaussians must have the same number of primitives"

    I = 0  # running total of the integral

    N_prim = len(cg1)  # number of primitives that each contracted gaussian has

    # mixing coefficients
    d1 = cg1.coefficients
    d2 = cg2.coefficients
    d3 = cg3.coefficients
    d4 = cg4.coefficients

    # normalization constants
    n1 = cg1.prefactors
    n2 = cg2.prefactors
    n3 = cg3.prefactors
    n4 = cg4.prefactors

    for i in range(N_prim):
        for j in range(N_prim):
            for k in range(N_prim):
                for l in range(N_prim):

                    alphas = [
                        cg1.alphas[i],
                        cg2.alphas[j],
                        cg3.alphas[k],
                        cg4.alphas[l],
                    ]

                    # have to be in Bohr units
                    centers = [
                        cg1.centers[i] * chemical.angstrom2bohr,
                        cg2.centers[j] * chemical.angstrom2bohr,
                        cg3.centers[k] * chemical.angstrom2bohr,
                        cg4.centers[l] * chemical.angstrom2bohr,
                    ]

                    inputs = alphas + centers

                    # unnormalized and unweighted by coefficients
                    result = unnormalized_primitive_gaussian_2e_integral(*inputs)

                    D = d1[i] * d2[j] * d3[k] * d4[l]
                    norm = n1[i] * n2[j] * n3[k] * n4[l]

                    I += D * norm * result

    return I


# dumb because it doesn't take advantage of symmetry of the integrals
def dumb_get_2e_integrals(cg_orbitals: List[ContractedGaussian]) -> torch.Tensor:
    """Evaluates all two electron integrals between contracted gaussians.

    Args:
        cg_orbitals: list of contracted gaussians

    Returns:
        outs: torch tensor of two-electron integrals.
              Entry [i, j, k, l] is the integral (ij|kl)
    """
    N = len(cg_orbitals)
    outs = torch.zeros((N, N, N, N))

    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    outs[i, j, k, l] = contracted_gaussian_2e_integral(
                        cg_orbitals[i], cg_orbitals[j], cg_orbitals[k], cg_orbitals[l]
                    )

    return outs


def contracted_gaussian_G_matrix(
    ee: torch.Tensor,
    P: torch.Tensor,
):
    """Compute the two-electron integral matrix (G) for a set of contracted gaussians.

    Args:
        cgee: two-electron integrals, computed from contracted gaussians
        P: Density matrix, constructed from MO coefficient matrix (C)

    Returns:
        G: torch tensor of the two-electron integral matrix
    """
    L = P.shape[0]

    G = torch.zeros((L, L))

    # loop over Gij indicies
    for i in range(L):  # i is mu
        for j in range(L):  # j is nu

            # loop over Pkl indices
            for k in range(L):  # k is lambda
                for l in range(L):  # l is sigma

                    G[i, j] += P[k, l] * (ee[i, j, l, k] - 0.5 * ee[i, k, l, j])

    return G
