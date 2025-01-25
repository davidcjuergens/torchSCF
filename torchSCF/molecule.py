"""Molecule class"""

import parsing
import dataclasses
import numpy as np 

class Molecule:
    """Basic container for molecular information"""


def __init__(self, xyz=None, atomic_numbers=None, basis_set=None):
    """Create a Molecule object
    
    """
    self.xyz = xyz
    self.atomic_numbers = atomic_numbers

