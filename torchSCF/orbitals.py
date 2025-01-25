# Classes for creating/manipulating orbital functions 
import torch
import torch.nn as nn 
import math 
from typing import Union
from tqdm import tqdm
import pdb 
import time 

ORIGIN = torch.tensor(0.0)

class Orbital(torch.nn.Module):
    # Base class for all orbitals
    def __init__(self): 
        super(Orbital, self).__init__()
    
    def evaluate_density(self, r):
        """
        Returns the probability density at r. 
        """
        return torch.tensor(float('nan')) # should be implemented by subclasses
    
    def forward(self, r:torch.tensor): 
        return self.evaluate_density(r)



class PrimitiveGaussian(Orbital): 
    """
    Gaussian Type orbital - SO eq. 3.203
    """

    def __init__(self,  alpha:  torch.tensor = torch.tensor(1.0), 
                        center: torch.tensor = ORIGIN.clone()):
        super().__init__()
        assert len(alpha.shape) in (0,1), f"Alpha must be a scalar: {alpha.shape=}" 
        assert len(center.shape) in (0,1), f"Center must be a scalar: {center.shape=}"
        
        self.alpha = nn.Parameter(alpha) 
        self.center = center

    def evaluate_density(self, r:torch.tensor): 
        """
        Returns probability density at r. 
        """
        prefactor = (2*self.alpha/math.pi)**(3/4)
        return prefactor * torch.exp(-self.alpha * (torch.abs(r-self.center)**2))
    

class ContractedGaussian(Orbital):
    """
    A Contracted Gaussian Function (i.e., linear combination of primitive gaussians)
    """

    def __init__(self, L, alphas:torch.tensor, ds:torch.tensor):
        """
        Parameters: 
            L: number of primitive gaussians
            alphas: list of alphas for each primitive gaussian
        """
        super().__init__()
        self.L = L
        self.ds = nn.Parameter(ds)

        self.primitive_gaussians = nn.ModuleList( [PrimitiveGaussian(alpha) for alpha in alphas] )

        # check if params are leaf nodes 
        assert all( [alpha.is_leaf for alpha in alphas] ), "Alphas must be leaf nodes"
        assert ds.is_leaf, "ds must be a leaf node"
        

    def evaluate_density(self, r:torch.tensor):
        """
        Returns the probability density at r, which is the sum of the primitive gaussians 
        evaluated at r. 
        """
        assert len(r.shape) >= 1, f"r must be a tensor: {r.shape=}"

        # should vectorize this...
        densities = torch.stack( [PG.evaluate_density(r) for PG in self.primitive_gaussians] )

        densities = densities * self.ds[:,None]

        return torch.sum(densities, axis=0)
        


            



        


