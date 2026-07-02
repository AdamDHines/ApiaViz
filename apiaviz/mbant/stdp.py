"""Anti-Hebbian STDP rule matching STDP.m exactly."""

import torch

from mbant.config import STDPParams


def stdp(delta_t: torch.Tensor, params: STDPParams = None) -> torch.Tensor:
    """Compute STDP weight change.

    Implements the anti-Hebbian STDP rule from STDP.m:
    - delta_t < 0: change = A_minus * exp(delta_t / tau_minus)
    - delta_t > 0: change = A_minus * exp(-delta_t / tau_plus)
    - delta_t == 0: change = 0

    Both cases produce negative (depressing) changes.

    Args:
        delta_t: Spike timing difference (t_pre - t_post).
        params: STDP parameters. Uses defaults if None.

    Returns:
        Weight change tensor (same shape as delta_t).
    """
    if params is None:
        params = STDPParams()

    change = torch.zeros_like(delta_t)

    neg_mask = delta_t < 0
    pos_mask = delta_t > 0

    change[neg_mask] = params.A_minus * torch.exp(
        delta_t[neg_mask] / params.tau_minus
    )
    change[pos_mask] = params.A_minus * torch.exp(
        -delta_t[pos_mask] / params.tau_plus
    )

    return change
