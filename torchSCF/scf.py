"""Hartree-Fock SCF calculations"""

import torch
import argparse

from torchSCF import molecule, parsers, integrals, linalg, observables

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

        if diff < ptol:
            return {"F": F, "P": P_out, "C": C_out}
        else:
            P = P_out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform a Hartree-Fock SCF calculation"
    )

    parser.add_argument("-xyz", type=str, help="XYZ file containing molecular geometry")
    parser.add_argument(
        "-basis_set",
        type=str,
        help="Basis set to use for the calculation",
        default="sto-3g",
    )
    parser.add_argument(
        "-sto_zeta",
        type=float,
        help="Zeta parameter for STO-NG basis set",
        default=1.24,
    )

    return parser.parse_args()

def scf_h2_energy(xyz, elements, basis_set, sto_zeta):
    """
    Computes the total energy of H2 using the SCF method
    """
    mol = molecule.Molecule(xyz, elements)
    mol.get_basis_set({"basis_set": basis_set, "sto_zeta": sto_zeta})
    L = len(mol.basis_set)

    S = integrals.compute_overlap_matrix(mol.basis_set)
    T = integrals.contracted_gaussian_T_matrix(mol.basis_set)
    V1 = integrals.contracted_gaussian_V_matrix(mol.basis_set, mol.xyz[0])
    V2 = integrals.contracted_gaussian_V_matrix(mol.basis_set, mol.xyz[1])
    V = V1 + V2
    Hcore = T + V
    ee = integrals.dumb_get_2e_integrals(mol.basis_set)

    P_init = torch.zeros((L, L))
    scf_results = density_matrix_scf(mol, P_init, ee, Hcore, S)

    # Convert from atomic basis to molecular orbital basis
    Hcore_mo, ee_mo = integrals.ao_to_mo(scf_results["C"], Hcore, ee)

    # Compute total energy
    E0, Etot = observables.compute_h2_energy(Hcore_mo, ee_mo, mol)

    return E0, Etot


def main():

    args = parse_args()

    parsed = parsers.parse_xyz(args.xyz, xyz_th=True)
    xyz = parsed["xyz"]
    elements = parsed["elements"]   
    basis_set = args.basis_set
    sto_zeta = args.sto_zeta

    E0, Etot = scf_h2_energy(xyz, elements, basis_set, sto_zeta)

    print(f"Total energy: {Etot.item()} Hartrees")
    print(f"Electronic energy: {E0.item()} Hartrees")

    


if __name__ == "__main__":
    main()
