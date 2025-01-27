"""Test SCF procedures for Szabo & Ostlund Chapter 3 problems"""

import os
import unittest
import torch 

from torchSCF import scf, parsers, molecule, integrals

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
        mol = molecule.Molecule(h2_data["xyz"], h2_data["elements"])

        maxiters = 100
        ptol = 1e-6
        basis_set = "sto-3g"
        sto_zeta = 1.24
        basis_set_kwargs = {"basis_set": basis_set, "sto_zeta": sto_zeta}

        # get basis set 
        mol.get_basis_set(basis_set_kwargs)

        alphas_golden = [0.168856, 0.623913, 3.42525]
        weights_golden = [0.444635, 0.535328, 0.154329]
        for i in range(3):
            self.assertAlmostEqual(mol.basis_set_params["alphas"][i], alphas_golden[i], places=6)
            self.assertAlmostEqual(mol.basis_set_params["weights"][i], weights_golden[i], places=6)
        

        # overlap matrix 
        S = integrals.compute_overlap_matrix(mol.basis_set)
        
        # from SO eq. 3.229
        S_golden = torch.tensor([[1.0, 0.6593], 
                                 [0.6593, 1.0]])

        torch.testing.assert_close(S, S_golden, rtol=1e-4, atol=1e-4)


        # Kinetic energy matrix
        T = integrals.contracted_gaussian_T_matrix(mol.basis_set)

        T_golden = torch.tensor([[0.7600, 0.2365],
                                 [0.2365, 0.7600]])
        
        torch.testing.assert_close(T, T_golden, rtol=1e-4, atol=1e-4)

        # nuclear attraction matrix -- first nucleus
        V1 = integrals.contracted_gaussian_V_matrix(mol.basis_set, mol.xyz[0])
        V1_golden = torch.tensor([[-1.2266, -0.5974],
                                  [-0.5974, -0.6538]])
        print(V1)
        torch.testing.assert_close(V1, V1_golden, rtol=1e-4, atol=1e-4)

        # nuclear attraction matrix -- second nucleus
        V2 = integrals.contracted_gaussian_V_matrix(mol.basis_set, mol.xyz[1])
        V2_golden = torch.tensor([[-0.6538, -0.5974],
                                  [-0.5974, -1.2266]])
        torch.testing.assert_close(V2, V2_golden, rtol=1e-4, atol=1e-4)

        # core Hamiltonian matrix
        Hcore = T + V1 + V2
        Hcore_golden = torch.tensor([[-1.1204, -0.9584],
                                     [-0.9584, -1.1204]])
        torch.testing.assert_close(Hcore, Hcore_golden, rtol=1e-4, atol=1e-4)
        