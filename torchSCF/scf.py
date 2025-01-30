"""Hartree-Fock SCF calculations"""

import torch
import argparse

from torchSCF import molecule, parsers, integrals, linalg

import pdb


def density_matrix_scf(
    mol: molecule.Molecule,
    P_init: torch.tensor,
    ee_integrals: torch.tensor,
    Hcore: torch.tensor,
    S: torch.tensor,
    maxiters: int = 100,
    ptol: float = 1e-6,
):
    """Perform SCF iterations until density matrix convergence.

    Following Szabo and Ostlund, page 146.

    Args:
        mol: Molecule object
        P_init: Initial guess for density matrix
        ee_integrals: Electron-electron integrals
        Hcore: Core Hamiltonian matrix
        S: Overlap matrix
        maxiters: Maximum number of iterations
        ptol: Density matrix tolerance
    """

    P = P_init

    for i in range(maxiters):

        G = integrals.contracted_gaussian_G_matrix(ee_integrals, P)

        F = Hcore + G

        Xcomplex = linalg.symmetric_orthogonalization(S)
        X = Xcomplex.real

        Fprime = X.T @ F @ X

        eps, Cprime = torch.linalg.eigh(Fprime)

        C_out = X @ Cprime
        P_out = linalg.c2p(C_out)

        diff = torch.norm(P_out - P)
        print(f"SCF iteration {i}, diff = {diff}")

        if diff < ptol:
            return {"F": F, "P": P_out, "C": C_out}
        else:
            P = P_out

    

def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform a Hartree-Fock SCF calculation"
    )

    parser.add_argument("-xyz", type=str, help="XYZ file containing molecular geometry")

    return parser.parse_args()


def main():

    args = parse_args()

    moldata = parsers.parse_xyz(args.xyz)


if __name__ == "__main__":
    main()
