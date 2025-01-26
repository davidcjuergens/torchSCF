"""Test SCF procedures for Szabo & Ostlund Chapter 3 problems"""

import os
import unittest

from torchSCF import scf, parsers, molecule

import pdb


class Test_STO_3G_H2(unittest.TestCase):
    """Run the STO-3G calculation for H2 from SO 3.5.2"""

    def setUp(self):
        """
        Resolve paths.
        """
        thisdir = os.path.dirname(os.path.abspath(__file__))
        golden_dir = os.path.join(thisdir, "goldens")
        self.h2_path = os.path.join(golden_dir, "h2.xyz")

    def test_scf(self):
        """
        Tests the SCF procedure for H2.
        """
        h2_data = parsers.parse_xyz(self.h2_path, xyz_th=True)
        h2 = molecule.Molecule(h2_data["xyz"], h2_data["elements"])

        maxiters = 100
        ptol = 1e-6
        basis_set = "sto-3g"

        inputs = {"mol": h2, "basis_set": basis_set, "maxiter": maxiters, "ptol": ptol}

        out = scf.run_scf(**inputs)
