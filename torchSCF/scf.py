"""Hartree-Fock SCF calculations"""

import argparse

import parsers
import molecule
import integrals


def compute_integrals(mol: molecule.Molecule):
    """

    Args:
        mol: molecule object

    Returns:
        S: overlap matrix
        Hcore: core Hamiltonian
        two_e_integrals: two electron integrals

    """
    


def run_scf(
    mol: molecule.Molecule,
    basis_set: str,
    maxiter: int = 100,
    etol: float = 1e-6,
    ptol: float = 1e-6,
):
    """Perform a Hartree-Fock SCF calculation.

    Following Szabo and Ostlund, page 146.

    Args:
        mol: molecule object
        basis_set: name of the basis set
        maxiter: maximum number of SCF iterations
        etol: tolerance for energy convergence (in Hartree)
        ptol: tolerance for density matrix convergence

    """
    # (1) Set up basis set
    mol.get_basis_set(basis_set)

    # (2) compute molecular integrals
    S = integrals.compute_overlap_matrix(mol.basis_set)
    print(S)
    return 
    Hcore = compute_core_hamiltonian(mol)
    two_e_integrals = compute_two_electron_integrals(mol)

    # (3) diagonalize overlap matrix S and get transformation matrix X
    Sdiag, X = get_canonical_orthogonalization(S)

    # (4) initial guess for density matrix
    P = get_initial_density(mol)

    for i in range(maxiter):

        # (5) Calculate G of eq. 3.154
        G = compute_G(P, two_e_integrals)

        # (6) Compute Fock matrix
        F = Hcore + G

        # (7) Calculate transformed Fock matrix
        Fprime = X.T @ F @ X

        # (8) Diagonalize Fprime to get Cprime and eps
        Cprime, eps = diagonalize(Fprime)

        # (9) Transform Cprime back to get C
        C = X @ Cprime

        # (10) Compute new density matrix
        P = compute_density(C)

        # (11) Compute electronic energy / determine if converged
        converged = evaluate_convergence(P, Pold, ptol)

        if converged:
            break

    return P, F, eps


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
