"""Hartree-Fock SCF calculations"""

import numpy as np
import torch
import argparse
import time

from torchSCF import molecule, parsers, integrals, linalg, observables, chemical

import pdb


def density_matrix_scf(
    mol: molecule.Molecule,
    P_init: torch.tensor,
    ee_integrals: torch.tensor,
    Hcore: torch.tensor,
    S: torch.tensor,
    maxiters: int = 100,
    ptol: float = 1e-6,
    alpha: float = 0.75,
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
        alpha: damping factor for Fock matrix (encourages convergence)
    """

    P = P_init
    diffs = []

    G = integrals.contracted_gaussian_G_matrix(ee_integrals, P_init)

    F = Hcore + G

    for i in range(maxiters):

        

        Xcomplex = linalg.symmetric_orthogonalization(S)
        X = Xcomplex.real

        Fprime = X.T @ F @ X

        eps, Cprime = torch.linalg.eigh(Fprime)

        C_out = X @ Cprime
        P_out = linalg.c2p(C_out)

        diff = torch.norm(P_out - P)
        diffs.append(diff)

        G = integrals.contracted_gaussian_G_matrix(ee_integrals, P)
        E = torch.trace(0.5*(2*Hcore+G) @ P_out)
        F = alpha*(Hcore + G) + (1-alpha)*F
        
        if diff < ptol:
            return {"F": F, "P": P_out, "C": C_out}
        else:
            P = P_out

        

    print("Returning unconverged values!!!")
    return {"F": F, "P": None, "C": C_out}


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
    parser.add_argument("-double", action="store_true", help="Use double precision")

    return parser.parse_args()


def scf_h2_energy(xyz, elements, basis_set, sto_zeta, maxiters=100, P_init=None):
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

    # if P_init is None:
    # P_init = torch.zeros((L, L))
    P_init = (1/np.sqrt(2))*torch.ones((L,L))

    scf_results = density_matrix_scf(mol, P_init, ee, Hcore, S, maxiters=maxiters)

    # Convert from atomic basis to molecular orbital basis
    Hcore_mo, ee_mo = integrals.ao_to_mo(scf_results["C"], Hcore, ee)

    # Compute total energy
    E0, Etot, nuc_rep = observables.compute_h2_energy(Hcore_mo, ee_mo, mol)

    return E0, Etot, scf_results["P"], nuc_rep


def H2_Etot_at_bond_length(bl: float = 0.740852, maxiters=1000, P_init=None):
    """Compute Etot for H2 at a given bond length

    Args:
        bl: bond length, in Angstroms
    """
    h2_path = "/home/davidcj/projects/torchSCF/tests/goldens/h2.xyz"
    parsed = parsers.parse_xyz(h2_path, xyz_th=True)
    xyz = parsed["xyz"]
    elements = parsed["elements"]

    original_bl = torch.norm(xyz[0] - xyz[1], p=2)
    diff = bl - original_bl
    xyz[1, -1] += diff  # change z coordinate of second H atom

    basis_set = "sto-3g"
    sto_zeta = 1.24

    start = time.time()
    E0, Etot, P, Nrep = scf_h2_energy(
        xyz, elements, basis_set, sto_zeta, maxiters=maxiters, P_init=P_init
    )
    print(f"SCF took {time.time() - start} seconds")

    return E0, Etot, P, Nrep


def h2_energy_landscape(fp, dmin=0.3, dmax=5, maxiters=100):
    """
    Compute h2 energy over various distances and safe to fp.
    """
    bls = np.linspace(dmin, dmax, num=100)

    E0s, Etots, Nreps = [], [], []
    P = None
    for bl in bls:
        E0, Etot, P, Nrep = H2_Etot_at_bond_length(bl, P_init=P)

        E0s.append(E0)
        Etots.append(Etot)
        Nreps.append(Nrep)

    with open(fp, "w") as f:
        f.write("bond_length,E0,Etot,Nrep\n")
        for i in range(len(bls)):
            to_write = f"{bls[i]},{E0s[i]},{Etots[i]},{Nreps[i]}\n"
            f.write(to_write)


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
    # main()

    # H2_Etot_at_bond_length(5)  # angstroms
    h2_energy_landscape("h2_energy_landscape_damped2_p_init_invsqrt2.csv", dmin=0.1, dmax=5, maxiters=1000)
