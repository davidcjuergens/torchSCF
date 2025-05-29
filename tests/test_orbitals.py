import unittest

from torchSCF import orbitals


class TestOrbitals(unittest.TestCase):

    def test_PrimitiveGaussian(self):
        alpha = 1.0
        center = (0.0, 0.0, 0.0)

        prim_gauss = orbitals.PrimitiveGaussian(alpha, center)

    def test_ContractedGaussian(self):
        L = 3
        alphas = [1.0, 1.0, 1.0]
        ds = [1.0, 1.0, 1.0]
        centers = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]

        contracted_gauss = orbitals.ContractedGaussian(alphas, ds, centers)
