"""Molecule class for storing molecular information"""

import torch
import numpy as np
import re
from typing import Optional, List

from torchSCF import chemical, orbitals
import pdb


class Molecule:
    """Basic container for molecular information"""

    def __init__(
        self, xyz: List[tuple], elements: List[str], n_electrons: Optional[int] = None
    ):
        """Create a Molecule object

        Args:
            xyz: list of atomic coordinates (in Angstroms!!!)
            elements: list of atomic symbols
            n_electrons: number of electrons in the molecule
        """
        self.xyz = xyz
        self.elements = elements

        if n_electrons is None:
            self.n_electrons = self.neutral_molecule_electrons()

    def neutral_molecule_electrons(self):
        """Count the number of electrons which would make the molecule neutral"""
        return sum([chemical.get_atomic_number(e) for e in self.elements])

    def get_basis_set(self, basis_set_kwargs: dict):
        """Get the basis set for the molecule

        Args:
            basis_set_kwargs: parameters for setting up basis set
        """
        basis_set = basis_set_kwargs["basis_set"]
        self.basis_set_params = chemical.basis_sets[basis_set.lower()]

        o = []
        if re.search(r"sto-.g", basis_set):
            # STO-NG basis set
            CG = orbitals.ContractedGaussian

            # get the Slater type orbital zeta, scale alphas accordingly
            zeta = basis_set_kwargs["sto_zeta"]
            self.basis_set_params = chemical.transform_sto_ng_zeta(
                zeta, self.basis_set_params
            )

            alphas = self.basis_set_params["alphas"]
            weights = self.basis_set_params["weights"]
            centers = self.xyz
            n_gaussians = len(alphas)  # the N in STO-NG

            for i in range(self.n_electrons):
                # create contracted gaussian for each electron
                cur_center = torch.stack([centers[i]] * n_gaussians)
                o.append(CG(alphas, weights, cur_center))
        else:
            raise NotImplementedError()

        self.basis_set = o

    @property
    def nuclear_repulsion(self):
        """Computes the nuclear repulsion energy"""
        return self.compute_nuclear_repulsion_energy()

    def compute_nuclear_repulsion_energy(self):
        """
        Computes simple coulombic repulsion between nuclei
        """

        # atomic numbers
        charges = torch.tensor([chemical.get_atomic_number(e) for e in self.elements])

        # compute distances between nuclei - don't double count
        dmap = torch.cdist(
            self.xyz * chemical.angstrom2bohr, self.xyz * chemical.angstrom2bohr
        )
        dmap = torch.triu(dmap, diagonal=1)
        triu = torch.triu_indices(dmap.shape[0], dmap.shape[1], offset=1)
        i, j = triu[0], triu[1]
        r = dmap[i, j]

        # compute repulsion (atomic units)
        repulsion = torch.sum(charges[i] * charges[j] / r)

        return repulsion
