"""Synapse models matching synapse.m, PN_KC_synapse.m, KC_EN_synapse.m."""

import torch

from mbant.config import PNKCSynapseParams, KCENSynapseParams, STDPParams
from mbant.stdp import stdp


def synapse_update(
    dt: float,
    S: torch.Tensor,
    spikes: torch.Tensor,
    phi_S: float,
    tau_syn_S: float,
) -> torch.Tensor:
    """Base synapse update matching synapse.m.

    S = S - S/tau_syn_S * dt + spikes * phi_S
    """
    return S - S / tau_syn_S * dt + spikes * phi_S


class PNKCSynapse:
    """Non-plastic PN->KC synapse (PN_KC_synapse.m).

    Wraps the base synapse with phi_S=0.93, tau_syn_S=3.0.
    State shape: (numPN, numKC).
    """

    def __init__(
        self,
        numPN: int,
        numKC: int,
        params: PNKCSynapseParams = None,
        device: torch.device = None,
    ):
        if params is None:
            params = PNKCSynapseParams()
        if device is None:
            device = torch.device("cpu")
        self.params = params
        self.device = device
        self.S = torch.zeros(numPN, numKC, dtype=torch.float32, device=device)

    def reset(self):
        self.S.zero_()

    def update(self, dt: float, spikes: torch.Tensor):
        """Update synaptic state.

        Args:
            spikes: Pre-synaptic spikes broadcast through connection matrix.
                    Shape (numPN, numKC).
        """
        self.S = synapse_update(
            dt, self.S, spikes, self.params.phi_S, self.params.tau_syn_S
        )


class KCENSynapse:
    """Plastic KC->EN synapse with three-factor learning (KC_EN_synapse.m).

    State variables:
        S: Synaptic conductance state (numKC, numEN)
        g: Weight matrix (numKC, numEN) — persists across images
        c: Eligibility trace / synaptic tag (numKC, numEN)
        d: Biogenic amine concentration (numKC, numEN)
    """

    def __init__(
        self,
        numKC: int,
        numEN: int,
        g_init: float,
        params: KCENSynapseParams = None,
        stdp_params: STDPParams = None,
        device: torch.device = None,
    ):
        if params is None:
            params = KCENSynapseParams()
        if stdp_params is None:
            stdp_params = STDPParams()
        if device is None:
            device = torch.device("cpu")

        self.params = params
        self.stdp_params = stdp_params
        self.device = device
        self.numKC = numKC
        self.numEN = numEN

        # Weight matrix — persists across images
        self.g = torch.full(
            (numKC, numEN), g_init, dtype=torch.float32, device=device
        )

        # Per-image state (reset each image)
        self.S = torch.zeros(numKC, numEN, dtype=torch.float32, device=device)
        self.c = torch.zeros(numKC, numEN, dtype=torch.float32, device=device)
        self.d = torch.zeros(numKC, numEN, dtype=torch.float32, device=device)

    def reset(self):
        """Reset per-image state (NOT weight matrix g)."""
        self.S.zero_()
        self.c.zero_()
        self.d.zero_()

    def update(
        self,
        dt: float,
        spikes: torch.Tensor,
        delta_t: torch.Tensor,
        pre_post_spike_occurred: torch.Tensor,
        BA: float,
    ):
        """Update synapse state with three-factor learning rule.

        Matches KC_EN_synapse.m exactly:
            S = synapse(dt, S, spikes, phi_S, tau_syn_S)
            dcdt = -c/tau_c + pre_post_spike * STDP(delta_t)
            c = c + dcdt * dt
            dddt = -d/tau_d
            d = d + dddt * dt + BA
            dgdt = c * d
            g = max(0.0001, g + dgdt * dt)

        Args:
            spikes: KC spike vector (numKC, 1) or (numKC, numEN).
            delta_t: t_spike_KC - t_spike_EN, shape (numKC, numEN).
            pre_post_spike_occurred: max(spike_KC, spike_EN), shape (numKC, numEN).
            BA: Biogenic amine release amount (scalar, 0 or 0.5).
        """
        p = self.params

        # Update synaptic conductance (base synapse model)
        self.S = synapse_update(dt, self.S, spikes, p.phi_S, p.tau_syn_S)

        # Eligibility trace update
        dcdt = -self.c / p.tau_c + pre_post_spike_occurred * stdp(
            delta_t, self.stdp_params
        )
        self.c = self.c + dcdt * dt

        # Biogenic amine concentration update
        dddt = -self.d / p.tau_d
        self.d = self.d + dddt * dt + BA

        # Weight update
        dgdt = self.c * self.d
        self.g = torch.clamp(self.g + dgdt * dt, min=0.0001)
