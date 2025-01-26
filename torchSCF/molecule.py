"""Molecule class for storing molecular information"""

import torch
import numpy as np
import re 
from typing import Optional, List

import chemical
import orbitals

import pdb

class Molecule:
    """Basic container for molecular information"""

    def __init__(
        self, xyz: List[tuple], elements: List[str], n_electrons: Optional[int] = None
    ):
        """Create a Molecule object"""
        self.xyz = xyz
        self.elements = elements

        if n_electrons is None:
            self.n_electrons = self.neutral_molecule_electrons()

    def neutral_molecule_electrons(self):
        """Count the number of electrons which would make the molecule neutral"""
        return sum([chemical.get_atomic_number(e) for e in self.elements])

    def get_basis_set(self, basis_set: str):
        """Get the basis set for the molecule

        Args:
            basis_set: name of the basis set
        """
        self.basis_set_params = chemical.basis_sets[basis_set.lower()]

        o = []
        if re.search(r"sto-.g", basis_set): # STO-NG basis set
            CG = orbitals.ContractedGaussian

            alphas = self.basis_set_params["alphas"]
            weights = self.basis_set_params["weights"]
            centers = self.xyz
            n_gaussians = len(alphas) # the N in STO-NG
            
            for i in range(self.n_electrons):
                # create contracted gaussian for each electron
                cur_center = torch.stack([centers[i]]*n_gaussians)
                o.append(CG(alphas, weights, cur_center))
        else:
            raise NotImplementedError()

        self.basis_set = o

            