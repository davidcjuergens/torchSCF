"""Linear algebra operations"""

import torch


def symmetric_orthogonalization(A: torch.Tensor) -> torch.Tensor:
    """Computes the inverse square root of a matrix using the symmetric orthogonalization method.

    Args:
        A : matrix to be orthogonalized

    Returns:
        X : orthogonalization matrix, of torch dtype torch.complex64
    """
    # (1) diagonalize A
    vals, vecs = torch.linalg.eig(A)

    # (2) form matrix with inverse square roots of eigenvalues on diagonal
    inv_s12 = torch.diag(1 / torch.sqrt(vals))

    # (3) form the transformation matrix X
    X = vecs @ inv_s12 @ vecs.T

    return X


def c2p(C: torch.Tensor, nocc: int) -> torch.Tensor:
    """Computes the density matrix from the coefficient matrix.

    SO equation 3.145.

    Args:
        C : MO coefficient matrix
        nocc: number of occupied orbitals

    Returns:
        P : density matrix
    """
    assert C.ndim == 2, "C must be a 2D tensor"
    C_occ = C[:, :nocc]
    P = 2 * C_occ @ C_occ.T
    return P
