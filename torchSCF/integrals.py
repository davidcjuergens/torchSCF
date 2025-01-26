"""Functions for evaluating integrals"""

import torch
import numpy as np
from typing import List, Union

from orbitals import PrimitiveGaussian, ContractedGaussian

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

    Ra = pg1.center
    Rb = pg2.center

    squared_distance = torch.sum((Ra - Rb) ** 2)

    K = torch.exp(-mu * squared_distance)

    overlap = ((np.pi / p) ** (3 / 2)) * K 

    # normalize 
    overlap *= pg1.prefactor * pg2.prefactor

    return overlap


def contracted_gaussian_overlap(cg1, cg2):
    """Compute single overlap S12 between two contracted gaussians."""
    assert len(cg1) == len(
        cg2
    ), "Contracted gaussians must have the same number of primitives"
    S12 = 0

    NG = len(cg1)
    d1 = cg1.coefficients
    d2 = cg2.coefficients

    for i in range(NG):
        for j in range(NG):
            prim_1 = cg1.primitives[i]
            prim_2 = cg2.primitives[j]

            S12 += d1[i] * d2[j] * primitive_gaussian_overlap(prim_1, prim_2) 

    return S12


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


def compute_overlap_matrix(basis_set: List):
    """Get the overlap matrix for a set of contracted gaussians.

    Args:
        basis_set: list of orbital objects
    """
    if all(
        [isinstance(o, ContractedGaussian) for o in basis_set]
    ):  # contracted gaussians
        return contracted_gaussian_overlap_matrix(basis_set)

    else:
        raise NotImplementedError()
