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


def c2p(C: torch.Tensor) -> torch.Tensor:
    """Computes the density matrix from the coefficient matrix.

    SO equation 3.145.

    NOTE: Not simply 2 * C @ C.T

    Args:
        C : coefficient matrix

    Returns:
        P : density matrix
    """

    P = torch.zeros_like(C)

    sum_to = C.shape[0] // 2  # sum to half the number of basis functions??

    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            for a in range(sum_to):
                P[i, j] += 2 * C[i, a] * C[j, a]

    return P
