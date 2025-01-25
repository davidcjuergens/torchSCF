"""Test SCF procedures for Szabo & Ostlund Chapter 3 problems"""

import unittest 

import scf 
import parsers

import pdb

class Test_STO_3G_H2(unittest.TestCase):
    """Run the STO-3G calculation for H2 from SO 3.5.2"""

    def test_scf(self):
        
        h2 = parsers.parse_xyz("h2.xyz")

        P, F, eps = scf.run_scf(h2)

