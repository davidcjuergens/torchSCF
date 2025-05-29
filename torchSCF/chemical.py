"""Basic chemical constants"""

from torchSCF.ptable import ALL_SYMBOLS


elements = ALL_SYMBOLS.split()
lower_elements = [e.lower() for e in elements]

# get atomic number for string representation of element
atomic_numbers = {e: i + 1 for i, e in enumerate(elements)}


def get_atomic_number(e):
    return atomic_numbers[e.strip()]


# Conversion of atomic units to SI units (Szabo and Ostlund, pg. 42)
a_0 = 5.2918e-11  # length: Bohr radius (meters)
m_e = 9.1095e-31  # mass: electron mass (kg)
e = 1.6022e-19  # charge: elementary charge (C)
eps_a = 4.3598e-18  # energy: Hartree (J)
hbar = 1.0546e-34  # angular momentum: reduced Planck constant (J*s)
ea_0 = e * a_0  # electric dipole moment (C*m)

# rename for clarity
bohr_radius = a_0
electron_mass = m_e
elementary_charge = e
hartree = eps_a

a_0_angstrom = a_0 * 1e10  # length: Bohr radius (Angstrom)
angstrom2bohr = 1 / a_0_angstrom
bohr2angstrom = a_0_angstrom

###################################
# Szabo Ostlund STO-NG basis sets #
###################################

# STO-NG with STO zeta = 1.0
sto_1g = {"alphas": [0.270950], "weights": [1.0]}
sto_2g = {"alphas": [0.151623, 0.851819], "weights": [0.678914, 0.430129]}
sto_3g = {
    "alphas": [0.109818, 0.405771, 2.22766],
    "weights": [0.444635, 0.535328, 0.154329],
}


def transform_sto_ng_zeta(zeta, sto):
    """Returns new STO-NG basis set params fit to slater zeta != 1.0

    Args:
        zeta (float): Slater exponent
        sto (dict): STO-NG basis set params

    Returns:
        dict: New STO-NG basis set params
    """
    return {"alphas": [zeta**2 * a for a in sto["alphas"]], "weights": sto["weights"]}


basis_sets = {"sto-1g": sto_1g, "sto-2g": sto_2g, "sto-3g": sto_3g}
