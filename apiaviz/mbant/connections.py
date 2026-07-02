"""Connection matrix generator matching connection_generator.m."""

import torch


def generate_connections(
    numPN: int = 360,
    numKC: int = 20000,
    numEN: int = 1,
    num_pn_per_kc: int = 10,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate connection matrices for PN-KC and KC-EN.

    Matches connection_generator.m:
        - PN->KC: Each KC receives exactly num_pn_per_kc randomly selected PNs.
          Dense binary matrix of shape (numPN, numKC).
        - KC->EN: All-to-all connectivity. Ones matrix of shape (numKC, numEN).

    Uses torch random state for reproducibility (controlled by torch.manual_seed).

    Args:
        numPN: Number of projection neurons.
        numKC: Number of Kenyon cells.
        numEN: Number of extrinsic neurons.
        num_pn_per_kc: Number of PN inputs per KC.
        device: Torch device.

    Returns:
        Tuple of (connection_PN_KC, connection_KC_EN).
    """
    if device is None:
        device = torch.device("cpu")

    # PN-KC connection: each KC receives num_pn_per_kc random PNs
    # Matches MATLAB: for each KC, randperm(numPN), take first 10
    connection_PN_KC = torch.zeros(
        numPN, numKC, dtype=torch.float32, device=device
    )
    for i_KC in range(numKC):
        pn_indices = torch.randperm(numPN, device=device)[:num_pn_per_kc]
        connection_PN_KC[pn_indices, i_KC] = 1.0

    # KC-EN connection: all-to-all
    connection_KC_EN = torch.ones(
        numKC, numEN, dtype=torch.float32, device=device
    )

    return connection_PN_KC, connection_KC_EN
