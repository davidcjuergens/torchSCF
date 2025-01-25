"""Basic chemical constants"""

from ptable import ALL_SYMBOLS


elements = ALL_SYMBOLS.split()
lower_elements = [e.lower() for e in elements]

# get atomic number for string representation of element
atomic_numbers = {e: i for i, e in enumerate(elements)}
get_atomic_number = lambda e: atomic_numbers[e.strip()]

# Conversion of atomic units to SI units (Szabo and Ostlund, pg. 42)
a_0   = 5.2918e-11 # length: Bohr radius (meters)
m_e   = 9.1095e-31 # mass: electron mass (kg)
e     = 1.6022e-19 # charge: elementary charge (C)
eps_a = 4.3598e-18 # energy: Hartree (J)
hbar  = 1.0546e-34 # angular momentum: reduced Planck constant (J*s)
ea_0  = e*a_0      # electric dipole moment (C*m)

# rename for clarity
bohr_radius = a_0
electron_mass = m_e
elementary_charge = e
hartree = eps_a

