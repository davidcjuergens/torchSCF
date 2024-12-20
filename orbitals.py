# Classes for creating/manipulating orbital functions 
import torch
import torch.nn as nn 
import math 
from typing import Union
from tqdm import tqdm

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


class STO(Orbital): 
    """
    Slater type orbital - SO eq. 3.202
    """

    def __init__(self, zeta     : torch.tensor = torch.tensor([1.0]), 
                       center   : torch.tensor = ORIGIN.clone()):
        """
        Constructs a slater orbital object. 

        Parameters: 
            zeta: 
        """
        super().__init__()
        self.zeta = zeta 
        self.center = center
        self.prefactor = (zeta**3/math.pi)**(0.5)

    def evaluate_density(self, r:torch.tensor): 
        """
        Returns probability density at r.
        """
        return self.prefactor*torch.exp(-self.zeta * torch.abs(r-self.center))
    

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
        


class STO_NG(Orbital): 
    """
    Contracted Gaussian Function built by fitting N Gaussian (NG) type orbitals 
    to a Slater Type Orbital (STO)
    """

    def __init__(self,  alphas,
                        ds, 
                        zeta : float            = 1.0, 
                        center                  = ORIGIN.clone(), 
                        contraction_length:int  = 3, 
                        ): 
        """
        Parameters: 
            zeta: STO zeta parameter 
            center: STO mean
            contraction_length: number of gaussians being used to approximate the STO
            alphas: list of alphas for each primitive gaussian
            ds: mixing parameter for each gaussian 
        """
        super().__init__()

        self.sto = STO(zeta, center) # slater orbital to fit the gaussians to
        self.contracted_gaussian = ContractedGaussian(contraction_length, alphas, ds)
    

    def fit(self, N=1000, lr=5e-2, dr=1e-4, rmin=0, rmax=10): 
        """
        Fits the contraction set of gaussians to the STO. 

        Parameters:
            N: number of iterations for the optimization algorithm
        """
        optimizer = torch.optim.SGD(self.contracted_gaussian.parameters(), lr=lr)

        # set of points to evaluate 
        r = torch.linspace(rmin, rmax, int( (rmax-rmin)/dr) )
        for i in tqdm( range(N) ): 
            optimizer.zero_grad()

            # compute overlap integral between STO and contracted gaussian
            p_sto   = self.sto.evaluate_density(r).detach()
            p_gauss = self.contracted_gaussian.evaluate_density(r)
            overlap = torch.trapz(p_sto*p_gauss, dx=dr)
            
            loss = - overlap

            loss.backward()
            optimizer.step()

            



        


