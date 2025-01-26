# Classes for creating/manipulating orbital functions
import torch
import torch.nn as nn
import math
from typing import Union, List
import pdb
import time

ORIGIN = torch.tensor(0.0)


class Orbital:
    # Base class for all orbitals
    def __init__(self):
        pass

    def evaluate_density(self, r):
        raise NotImplementedError

    def torchify(self):
        pass


class PrimitiveGaussian(Orbital):
    """
    Gaussian Type orbital - SO eq. 3.203
    """

    def __init__(
        self,
        alpha: float,
        center: tuple,
    ):
        """Basic container for primitive gaussian type orbital.

        Args:
            alpha: exponential factor
            center: center of the gaussian
        """

        self.alpha = alpha
        self.center = torch.tensor(center)

    def evaluate_density(self, r: torch.tensor):
        """
        Returns probability density at r.
        """
        prefactor = (2 * self.alpha / math.pi) ** (3 / 4)
        return prefactor * torch.exp(-self.alpha * (torch.abs(r - self.center) ** 2))
    
    @ property 
    def prefactor(self):
        return (2 * self.alpha / math.pi) ** (3 / 4)

    def __len__(self):
        return 1


class ContractedGaussian(Orbital):
    """A linear combination of primitive gaussians"""

    def __init__(
        self,
        alphas: List[float],
        coefficients: List[float],
        centers: List[Union[tuple, torch.tensor]],
    ):
        """Create a contracted gaussian.

        Args:
            alphas: list of exponents
            centers: list of centers
            coefficients: list of mixture coefficients
        """

        self.primitives = [
            PrimitiveGaussian(alpha, center) for alpha, center in zip(alphas, centers)
        ]

        self.coefficients = torch.tensor(coefficients)
        self.alphas = torch.tensor(alphas)

        if (not torch.is_tensor(centers)) and (torch.is_tensor(centers[0])):
            self.centers = torch.stack(centers)

    def __len__(self):
        return len(self.primitives)
