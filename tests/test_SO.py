"""Test SCF procedures for Szabo & Ostlund Chapter 3 problems"""

import os
import unittest

import torch

from torchSCF import parsers, molecule, integrals, linalg, scf

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

        self.basis_set = "sto-3g"
        self.sto_zeta = 1.24

    def test_parse_and_mol_and_basis_set(self):
        """
        Test the construction and basis-set getting of Molecule.
        """
        h2_data = parsers.parse_xyz(self.h2_path, xyz_th=True)
        mol = molecule.Molecule(h2_data["xyz"], h2_data["elements"])
        self.assertEqual(mol.n_electrons, 2)

        basis_set_kwargs = {"basis_set": self.basis_set, "sto_zeta": self.sto_zeta}

        # get basis set
        mol.get_basis_set(basis_set_kwargs)

        alphas_golden = [0.168856, 0.623913, 3.42525]
        weights_golden = [0.444635, 0.535328, 0.154329]
        for i in range(3):
            self.assertAlmostEqual(
                mol.basis_set_params["alphas"][i], alphas_golden[i], places=6
            )
            self.assertAlmostEqual(
                mol.basis_set_params["weights"][i], weights_golden[i], places=6
            )

    def test_minimal_basis_h2(self):
        """
        Tests the SCF procedure for H2.
        """
        h2_data = parsers.parse_xyz(self.h2_path, xyz_th=True)
        mol = molecule.Molecule(h2_data["xyz"], h2_data["elements"])

        basis_set = "sto-3g"
        sto_zeta = 1.24

        basis_set_kwargs = {"basis_set": basis_set, "sto_zeta": sto_zeta}

        # get basis set
        mol.get_basis_set(basis_set_kwargs)

        alphas_golden = [0.168856, 0.623913, 3.42525]
        weights_golden = [0.444635, 0.535328, 0.154329]
        for i in range(3):
            self.assertAlmostEqual(
                mol.basis_set_params["alphas"][i], alphas_golden[i], places=6
            )
            self.assertAlmostEqual(
                mol.basis_set_params["weights"][i], weights_golden[i], places=6
            )

        # overlap matrix
        S = integrals.compute_overlap_matrix(mol.basis_set)

        # from SO eq. 3.229
        S_golden = torch.tensor([[1.0, 0.6593], [0.6593, 1.0]])

        torch.testing.assert_close(S, S_golden, rtol=1e-4, atol=1e-4)

        # Kinetic energy matrix
        T = integrals.contracted_gaussian_T_matrix(mol.basis_set)

        T_golden = torch.tensor([[0.7600, 0.2365], [0.2365, 0.7600]])

        torch.testing.assert_close(T, T_golden, rtol=1e-4, atol=1e-4)

        # nuclear attraction matrix -- first nucleus
        V1 = integrals.contracted_gaussian_V_matrix(mol.basis_set, mol.xyz[0])
        V1_golden = torch.tensor([[-1.2266, -0.5974], [-0.5974, -0.6538]])

        torch.testing.assert_close(V1, V1_golden, rtol=1e-4, atol=1e-4)

        # nuclear attraction matrix -- second nucleus
        V2 = integrals.contracted_gaussian_V_matrix(mol.basis_set, mol.xyz[1])
        V2_golden = torch.tensor([[-0.6538, -0.5974], [-0.5974, -1.2266]])
        torch.testing.assert_close(V2, V2_golden, rtol=1e-4, atol=1e-4)

        # core Hamiltonian matrix
        Hcore = T + V1 + V2
        Hcore_golden = torch.tensor([[-1.1204, -0.9584], [-0.9584, -1.1204]])
        torch.testing.assert_close(Hcore, Hcore_golden, rtol=1e-4, atol=1e-4)

        # initial guess at coefficient matrix C
        C_init = torch.ones((2, 2)) * 0.5
        P_init = linalg.c2p(C_init)

        s12 = S_golden[0, 1]
        c11 = 1 / (torch.sqrt(2 * (1 + s12)))
        c12 = 1 / (torch.sqrt(2 * (1 - s12)))

        C_golden = torch.tensor([[c11, c12], [c11, -c12]])
        P_golden = torch.ones((2, 2)) * (1 / (1 + s12))
        P_from_Cgolden = linalg.c2p(C_golden)
        self.assertTrue(torch.allclose(P_golden, P_from_Cgolden))

        # two electron integrals
        ee = integrals.dumb_get_2e_integrals(mol.basis_set)

        # F11 computed according to SO excercise 3.25
        F11_manual = (1 / (1 + S[0, 1])) * (
            0.5 * ee[0, 0, 0, 0]
            + ee[0, 0, 1, 1]
            + ee[0, 0, 0, 1]
            - 0.5 * ee[0, 1, 0, 1]
        ) + Hcore[0, 0]
        F11_golden = -0.3655
        self.assertAlmostEqual(F11_manual.item(), F11_golden, places=4)

        # F12 computed according to SO excercise 3.25
        F12_manual = (1 / (1 + S[0, 1])) * (
            -0.5 * ee[0, 0, 1, 1] + ee[0, 0, 0, 1] + 1.5 * ee[0, 1, 0, 1]
        ) + Hcore[0, 1]
        F12_golden = -0.5939
        self.assertAlmostEqual(F12_manual.item(), F12_golden, places=4)

        # two electron integral / molecular orbital matrix
        G = integrals.contracted_gaussian_G_matrix(ee, P_golden)

        # full hamiltonian matrix (Fock matrix)
        F = Hcore + G

        # compare the automated Fock matrix with golden/manual calculations
        self.assertAlmostEqual(F[0, 0].item(), F11_golden, places=4)
        self.assertAlmostEqual(F[0, 1].item(), F12_golden, places=4)

        # Fock should be symmetric in RHF H2 calculation
        self.assertTrue(torch.allclose(F, F.T))

        # get transformation matrix X
        Xcomplex = linalg.symmetric_orthogonalization(S)
        X, Ximag = Xcomplex.real, Xcomplex.imag

        # hopefully imaginary part is zero
        self.assertTrue(torch.allclose(Ximag, torch.zeros_like(Ximag)))

        # transform Fock matrix
        Fprime = X.T @ F @ X

        # diagonalize Fprime
        eps, Cprime = torch.linalg.eigh(Fprime)

        # transform Cprime back to original basis
        C_out = X @ Cprime
        P_out = 2 * C_out @ C_out.T

        # Electron ground state energy of H2!
        E0 = (F[0, 0] + Hcore[0, 0] + F[0, 1] + Hcore[0, 1]) / (1 + S[0, 1])
        E0_golden = -1.8310
        self.assertAlmostEqual(E0.item(), E0_golden, places=4)

        # Get E total with nuclear repulsion
        E_total = E0 + mol.nuclear_repulsion
        E_total_golden = -1.1167
        self.assertAlmostEqual(E_total.item(), E_total_golden, places=4)

        # Perform SCF iterations
        maxiters = 100
        ptol = 1e-6
        scf_out = scf.density_matrix_scf(
            mol, P_init, ee, Hcore, S, maxiters=maxiters, ptol=ptol
        )

        # Compare SCF output with golden values
        Fscf = scf_out["F"]
        self.assertTrue(torch.allclose(Fscf, F, rtol=1e-4, atol=1e-4))
        E0scf = (Fscf[0, 0] + Hcore[0, 0] + Fscf[0, 1] + Hcore[0, 1]) / (1 + S[0, 1])
        self.assertAlmostEqual(E0scf.item(), E0_golden, places=4)

        # Convert from atomic basis to molecular orbital basis
        Hcore_mo, ee_mo = integrals.ao_to_mo(C_out, Hcore, ee)

        # test one-electron integrals after transformation
        h11 = Hcore_mo[0, 0]
        h22 = Hcore_mo[1, 1]
        h11_golden = -1.2528
        h22_golden = -0.4756
        self.assertAlmostEqual(h11.item(), h11_golden, places=4)
        self.assertAlmostEqual(h22.item(), h22_golden, places=4)

        # test two-electron integrals after transformation
        J11 = ee_mo[0, 0, 0, 0]
        J12 = ee_mo[0, 0, 1, 1]
        J22 = ee_mo[1, 1, 1, 1]
        K12 = ee_mo[0, 1, 1, 0]
        J11_golden = 0.6746
        J12_golden = 0.6636
        J22_golden = 0.6975
        K12_golden = 0.1813

        self.assertAlmostEqual(J11.item(), J11_golden, places=4)
        self.assertAlmostEqual(J12.item(), J12_golden, places=4)
        self.assertAlmostEqual(J22.item(), J22_golden, places=4)
        self.assertAlmostEqual(K12.item(), K12_golden, places=4)

        # Compute energies in MO basis
        E0_from_mo = 2 * h11 + J11
        self.assertAlmostEqual(E0_from_mo.item(), E0_golden, places=4)
