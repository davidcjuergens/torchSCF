import torch 
import unittest 
import pdb

import orbitals


class TestOrbitals(unittest.TestCase):

    def test_PrimitiveGaussian(self):
        alpha = torch.tensor([1.0])
        center = torch.tensor([0.0])

        prim_gauss = orbitals.PrimitiveGaussian(alpha=alpha, center=center)
        r = torch.tensor([0.0])
        p1 = prim_gauss.evaluate_density(r)
        self.assertEqual(type(p1), torch.Tensor)

        r2 = torch.linspace(0, 10, 100)
        p2 = prim_gauss.evaluate_density(r2)
        self.assertEqual(type(p2), torch.Tensor)

    def test_STO(self): 
        zeta = torch.tensor([1.0])
        center = torch.tensor([0.0])

        sto = orbitals.STO(zeta=zeta, center=center)
        r = torch.tensor(0.0)
        p1 = sto.evaluate_density(r)
        self.assertEqual(type(p1), torch.Tensor)

        r2 = torch.linspace(0, 10, 100)
        p2 = sto.evaluate_density(r2)
        self.assertEqual(type(p2), torch.Tensor)

    def test_ContractedGaussian(self):
        L = 3
        alphas = torch.tensor([1,1,1]).float()
        ds = torch.tensor([1/3,1/3,1/3])

        contracted_gauss = orbitals.ContractedGaussian(L, alphas, ds)

        # r = torch.tensor(0.0)
        # p1 = contracted_gauss.evaluate_density(r)
        # self.assertEqual(type(p1), torch.Tensor)
        
        r2 = torch.linspace(0, 10, 100)
        p2 = contracted_gauss.evaluate_density(r2)
        self.assertEqual(type(p2), torch.Tensor)

    def test_STO_NG(self): 
        """
        Tests the STO_NG class. 
        """
    
        alphas = torch.tensor([1., 1., 1.])
        ds = torch.tensor([1/3, 1/3, 1/3])

        sto_ng = orbitals.STO_NG(alphas, ds)

        sto_ng.fit(N=3)