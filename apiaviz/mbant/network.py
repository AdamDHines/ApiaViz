"""Full network simulation loop matching GPU_network_fun.m."""

import torch

from mbant.config import NetworkConfig
from mbant.neurons import PNLayer, KCLayer, ENLayer
from mbant.synapses import PNKCSynapse, KCENSynapse
from mbant.connections import generate_connections


class MushroomBodyNetwork:
    """Mushroom body spiking neural network.

    Implements the complete simulation from GPU_network_fun.m:
    - 360 PNs -> 20,000 KCs -> 1 EN
    - Conductance-based synapses
    - Three-factor learning (STDP + biogenic amine) on KC->EN

    The weight matrix (KC->EN) persists across image presentations.
    All other state is reset for each new image.
    """

    def __init__(
        self,
        config: NetworkConfig = None,
        device: torch.device = None,
    ):
        if config is None:
            config = NetworkConfig()
        if device is None:
            device = torch.device("cpu")

        self.config = config
        self.device = device

        # Generate connection matrices
        self.connection_PN_KC, self.connection_KC_EN = generate_connections(
            config.numPN,
            config.numKC,
            config.numEN,
            config.num_pn_per_kc,
            device,
        )

        # Create neuron layers
        self.pn = PNLayer(config.numPN, config.pn_params, device)
        self.kc = KCLayer(config.numKC, config.kc_params, device)
        self.en = ENLayer(config.numEN, config.en_params, device)

        # Create synapses
        self.syn_pn_kc = PNKCSynapse(
            config.numPN, config.numKC, config.pn_kc_synapse_params, device
        )
        self.syn_kc_en = KCENSynapse(
            config.numKC,
            config.numEN,
            config.g_KC_EN,  # initial weight
            config.kc_en_synapse_params,
            config.stdp_params,
            device,
        )

        # Null input for post-stimulus period
        self.null_input = torch.zeros(
            config.numPN, dtype=torch.float32, device=device
        )

    @property
    def weight_matrix_KC_EN(self) -> torch.Tensor:
        """Access the KC->EN weight matrix."""
        return self.syn_kc_en.g

    @weight_matrix_KC_EN.setter
    def weight_matrix_KC_EN(self, value: torch.Tensor):
        """Set the KC->EN weight matrix."""
        self.syn_kc_en.g = value.to(self.device)

    def simulate(
        self, input_current: torch.Tensor, reward: int
    ) -> dict:
        """Run network simulation for one image presentation.

        Matches GPU_network_fun.m: 50ms simulation at 1ms timesteps.
        All state is reset except weight_matrix_KC_EN.

        Args:
            input_current: PN input current, shape (numPN,).
            reward: 0 or 1. If 1, BA=0.5 released at t=40ms.

        Returns:
            Dictionary with:
                - spike_time_PN: (numPN, interval) spike raster
                - spike_time_KC: (numKC, interval) spike raster
                - spike_time_EN: (numEN, interval) spike raster
        """
        cfg = self.config
        dt = cfg.dt
        interval = cfg.interval
        num_steps = int(interval / dt)

        # Reset all per-image state (not weights)
        self.pn.reset()
        self.kc.reset()
        self.en.reset()
        self.syn_pn_kc.reset()
        # Reset synapse per-image state (S, c, d) but NOT weight g
        self.syn_kc_en.S.zero_()
        self.syn_kc_en.c.zero_()
        self.syn_kc_en.d.zero_()

        # Spike recording matrices
        spike_time_PN = torch.zeros(
            cfg.numPN, num_steps, dtype=torch.float32, device=self.device
        )
        spike_time_KC = torch.zeros(
            cfg.numKC, num_steps, dtype=torch.float32, device=self.device
        )
        spike_time_EN = torch.zeros(
            cfg.numEN, num_steps, dtype=torch.float32, device=self.device
        )

        for idt in range(num_steps):
            t = (idt + 1) * dt  # MATLAB: t = idt*dt where idt starts at 1
            BA = 0.0

            # Input current and reward signal
            # MATLAB: if t < 40.01, I_PN = input
            if t < cfg.ba_release_time + 0.01:
                I_PN = input_current
                if t == cfg.ba_release_time and reward == 1:
                    BA = cfg.ba_magnitude
            else:
                I_PN = self.null_input

            # Update PN->KC synapses
            # MATLAB: PN_spikes = bsxfun(@times, connection_PN_KC, spike_PN)
            # spike_PN is (numPN,), connection_PN_KC is (numPN, numKC)
            # Broadcasting: spike_PN[:, None] * connection_PN_KC
            pn_spikes_broadcast = self.pn.spike.unsqueeze(1) * self.connection_PN_KC
            self.syn_pn_kc.update(dt, pn_spikes_broadcast)

            # Update KC->EN synapses
            # MATLAB: pre_post_spike_occured = bsxfun(@max, spike_KC, spike_EN)
            pre_post_spike_occurred = torch.max(
                self.kc.spike.unsqueeze(1),
                self.en.spike.unsqueeze(0),
            )
            # MATLAB: delta_t = t_spike_KC - t_spike_EN
            delta_t = self.kc.t_spike.unsqueeze(1) - self.en.t_spike.unsqueeze(0)

            self.syn_kc_en.update(
                dt,
                self.kc.spike.unsqueeze(1),  # (numKC, 1)
                delta_t,
                pre_post_spike_occurred,
                BA,
            )

            # Compute post-synaptic currents
            # MATLAB: I_KC = sum(g_PN_KC * bsxfun(@times, synapses_PN_KC, (0-KC(:,1))'))';
            # This is: for each KC j, I_KC(j) = g_PN_KC * sum_i(S_PN_KC(i,j)) * (0 - V_KC(j))
            I_KC = cfg.g_PN_KC * torch.sum(self.syn_pn_kc.S, dim=0) * (
                0.0 - self.kc.v
            )

            # MATLAB: I_EN = weight_matrix_KC_EN' * synapses_KC_EN * (0 - EN(1))
            # weight_matrix_KC_EN is (numKC, numEN), synapses_KC_EN is (numKC, numEN)
            # For numEN=1: sum(g * S) * (0 - V_EN)
            I_EN = (
                torch.sum(self.syn_kc_en.g * self.syn_kc_en.S, dim=0)
                * (0.0 - self.en.v)
            )

            # Update neurons
            self.pn.step(dt, t, I_PN)
            spike_time_PN[:, idt] = self.pn.spike

            self.kc.step(dt, t, I_KC)
            spike_time_KC[:, idt] = self.kc.spike

            self.en.step(dt, t, I_EN)
            spike_time_EN[:, idt] = self.en.spike

        return {
            "spike_time_PN": spike_time_PN,
            "spike_time_KC": spike_time_KC,
            "spike_time_EN": spike_time_EN,
        }
