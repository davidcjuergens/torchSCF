"""Compute observables"""

import torch 

from torchSCF import molecule

def compute_h2_energy(Hcore_mo: torch.tensor, ee_mo: torch.tensor, mol: molecule.Molecule):
    """Compute the total energy of the molecule

    Args:
        Hcore_mo: core Hamiltonian matrix in molecular orbital basis
        ee_mo: two-electron integrals in molecular orbital basis
    """
    

    h11 = Hcore_mo[0, 0]
    h22 = Hcore_mo[1, 1]

    J11 = ee_mo[0, 0, 0, 0]
    J12 = ee_mo[0, 0, 1, 1]
    J22 = ee_mo[1, 1, 1, 1]
    K12 = ee_mo[0, 1, 1, 0]

    E0 = 2 * h11 + J11
    Nrep = mol.nuclear_repulsion

    Etot = E0 + Nrep

    return E0, Etot, Nrep