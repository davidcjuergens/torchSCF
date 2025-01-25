"""Basic chemical constants"""

from ptable import ALL_SYMBOLS


elements = ALL_SYMBOLS.split()
lower_elements = [e.lower() for e in elements]

# get atomic number for string representation of element
atomic_numbers = {e: i for i, e in enumerate(elements)}
get_atomic_number = lambda e: atomic_numbers[e.strip()]