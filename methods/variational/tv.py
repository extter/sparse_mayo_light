"""
tv.py

Utilities per ricostruzione variazionale TV nel task Sparse-view CT.
"""


import numpy as np
import torch

from third_party.ippy import operators, solvers


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_operators_dict(angle_configs, image_size, det_size):
    """
    Crea un dizionario di operatori CTProjector, uno per ogni configurazione angolare.

    """
    operators_dict = {}

    for n_angles in angle_configs:
        projector = operators.CTProjector(
            img_shape=(image_size, image_size),
            angles=np.linspace(0, np.pi, n_angles, endpoint=False),
            det_size=det_size,
            geometry="parallel",
            force_cpu=False,
        )
        operators_dict[n_angles] = projector

    return operators_dict


def create_tv_solver(projector):
    return solvers.ChambollePockTpVUnconstrained(projector)


def reconstruct_tv(
    y_delta,
    solver,
    lmbda,
    maxiter,
    p,
    x_true=None,
    starting_point=None,
    verbose=False,
):
    """
    Esegue la ricostruzione TV da un sinogramma corrotto.

    Returns:
        tuple[torch.Tensor, dict]: soluzione ricostruita e info del solver
    """
    x_sol, info = solver(
        y_delta,
        lmbda=lmbda,
        starting_point=starting_point,
        x_true=x_true,
        maxiter=maxiter,
        p=p,
        verbose=verbose,
    )
    return x_sol, info


def reconstruct_and_save_numpy(
    y_delta,
    solver,
    lmbda,
    maxiter,
    p,
    device,
    x_true=None,
    starting_point=None,
    verbose=False,
):
    """
    Ricostruisce e restituisce direttamente l'output come array NumPy 2D.

    Returns:
        np.ndarray: ricostruzione shape (H, W)
    """
    y_delta = y_delta.to(device)

    if x_true is not None:
        x_true = x_true.to(device)

    x_sol, _ = reconstruct_tv(
        y_delta=y_delta,
        solver=solver,
        lmbda=lmbda,
        maxiter=maxiter,
        p=p,
        x_true=x_true,
        starting_point=starting_point,
        verbose=verbose,
    )

    return x_sol.detach().cpu().squeeze().numpy().astype(np.float32)