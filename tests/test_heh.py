"""Test for HeH+ system from Szabo Ostlund"""

import os
from omegaconf import OmegaConf

import torch

from torchSCF import parsers
from torchSCF import molecule
from torchSCF import integrals
from torchSCF import linalg


def get_heh_conf():
    # wherever tests/ is
    this_folder = os.path.dirname(os.path.abspath(__file__))
    # pkg root is tests/../
    pkg_root = os.path.dirname(this_folder)

    yml = os.path.join(pkg_root, "torchSCF/config", "so_rhf_heh.yaml")
    # load the config
    conf = OmegaConf.load(yml)

    return conf


def test_heh():
    """Run the Szabo Ostlund HeH+ calculation"""
    conf = get_heh_conf()
    parsed = parsers.parse_xyz(lines=conf.xyz, xyz_th=True)
    mol = molecule.Molecule(
        parsed["xyz"], parsed["elements"], net_charge=conf.net_charge
    )

    # the order of zetas must match the order of electron centers
    bs_kwargs = {
        "basis_set": conf.basis_set.type,  # "sto-3g",
        "sto_zetas": conf.basis_set.sto_zetas,  # [2.0925, 1.24],
    }

    mol.get_basis_set(**bs_kwargs)

    S = integrals.compute_overlap_matrix(mol.basis_set)
    S_golden = torch.tensor([[1.0, 0.4508], [0.4508, 1.0]])
    assert torch.allclose(S, S_golden, atol=1e-4)

    T = integrals.contracted_gaussian_T_matrix(mol.basis_set)
    T_golden = torch.tensor([[2.1643, 0.1670], [0.1670, 0.7600]])
    assert torch.allclose(T, T_golden, atol=1e-4)

    Z = mol.atomic_numbers
    V1 = integrals.contracted_gaussian_V_matrix(mol.basis_set, mol.xyz[0], Z[0])
    V2 = integrals.contracted_gaussian_V_matrix(mol.basis_set, mol.xyz[1], Z[1])

    V1_golden = torch.tensor([[-4.1398, -1.1029], [-1.1029, -1.2652]])
    V2_golden = torch.tensor([[-0.6772, -0.4113], [-0.4113, -1.2266]])

    torch.testing.assert_close(V1, V1_golden, atol=5e-5, rtol=4e-5)
    torch.testing.assert_close(V2, V2_golden, atol=5e-5, rtol=4e-5)

    Hcore = T + V1 + V2
    Hcore_golden = torch.tensor([[-2.6527, -1.3472], [-1.3472, -1.7318]])
    torch.testing.assert_close(Hcore, Hcore_golden, atol=5e-5, rtol=4e-5)

    ee = integrals.dumb_get_2e_integrals(mol.basis_set)
    torch.testing.assert_close(ee[0, 0, 0, 0].item(), 1.3072, atol=5e-5, rtol=4e-5)
    torch.testing.assert_close(ee[1, 0, 0, 0].item(), 0.4373, atol=5e-5, rtol=4e-5)
    torch.testing.assert_close(ee[1, 0, 1, 0].item(), 0.1773, atol=5e-5, rtol=4e-5)
    torch.testing.assert_close(ee[1, 1, 0, 0].item(), 0.6057, atol=5e-5, rtol=4e-5)
    torch.testing.assert_close(ee[1, 1, 1, 0].item(), 0.3118, atol=5e-5, rtol=4e-5)
    torch.testing.assert_close(ee[1, 1, 1, 1].item(), 0.7746, atol=5e-5, rtol=4e-5)

    Xcomplex = linalg.symmetric_orthogonalization(S)
    X = Xcomplex.real

    ### now perform SCF

    F = Hcore.clone()  # initial guess for Fock matrix
    Fprime = X.T @ F @ X

    # SCF iterations
    for i in range(6):
        eps, Cprime = torch.linalg.eigh(Fprime)
        C = X @ Cprime
        P = linalg.c2p(C, nocc=1)  # 1 occupied orbital = 2 electrons
        G = integrals.contracted_gaussian_G_matrix(ee, P)
        F = Hcore + G
        Fprime = X.T @ F @ X

    Cfinal_golden = torch.tensor([[0.8019, -0.7823], [0.3368, 1.0684]])
    # for testing purposes, check if the sign is correct on output C
    # has no effect on energies
    if C[0,0] < 0:
        C = -C
    torch.testing.assert_close(C, Cfinal_golden, atol=5e-5, rtol=4e-5)
    eps_golden = torch.tensor([-1.5975, -0.0617]) # orbital energies
    torch.testing.assert_close(eps, eps_golden, atol=5e-5, rtol=4e-5)