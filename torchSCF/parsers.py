"""File parsing"""

import numpy as np
import torch
from typing import Optional


def parse_xyz(
    filename: Optional[str] = None, xyz_th: bool = False, lines: Optional[list] = None
):
    """Parse an xyz file

    XYZ file format:

        -----------------
        <number of atoms>
        <comment line>
        <atom1> <x> <y> <z>
        <atom2> <x> <y> <z>
        ...
        -----------------

    Args:
        filename: The name of the file to parse
        xyz_th: Return xyz coordinates as a torch tensor (else, numpy array)
        lines: if provided, use these lines instead of reading from the file
    Returns:
        dict: Dictionary containing
    """
    have_filename = filename is not None
    have_lines = lines is not None
    assert have_filename ^ have_lines, "Provide either filename or lines, not both"
    elements, xyz = [], []

    if filename is not None:
        with open(filename, "r") as f:
            lines = f.readlines()
    else:
        pass  # lines is already provided

    for i, line in enumerate(lines):
        if i == 0:
            natoms = int(line)  # number of atoms
        elif i == 1:
            comment = line  # comment line
        else:
            element, *coords = line.strip().split()
            elements.append(element)
            xyz.append([float(coord) for coord in coords])

    # sanity checks
    assert len(elements) == natoms, (
        f".xyz file says {natoms} atoms, but found {len(elements)} atoms"
    )

    xyz = np.array(xyz) if not xyz_th else torch.tensor(xyz)

    return {"natoms": natoms, "comment": comment, "elements": elements, "xyz": xyz}
