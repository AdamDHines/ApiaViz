"""Izhikevich neuron models matching PN_neuron.m, KC_neuron.m, EN_neuron.m."""

import torch

from mbant.config import PNParams, KCParams, ENParams


class IzhikevichLayer:
    """Vectorized Izhikevich neuron layer.

    Implements the exact same equations as the MATLAB neuron functions:
        if v > v_t:
            v = c; u = u + d; spike = 1
        epsilon = epsilon_std * randn
        v = v + dt/2 * (k*(v-v_r)*(v-v_t) - u + I + epsilon) / C  (applied twice)
        u = u + dt * a * (b*(v-v_r) - u)

    Note: epsilon is sampled once and reused for both half-step voltage updates,
    matching the MATLAB implementation exactly.
    """

    def __init__(
        self,
        num_neurons: int,
        params,
        device: torch.device = None,
        track_spike_time: bool = False,
    ):
        if device is None:
            device = torch.device("cpu")
        self.num_neurons = num_neurons
        self.params = params
        self.device = device
        self.track_spike_time = track_spike_time

        # State variables
        self.v = torch.full(
            (num_neurons,), params.v_init, dtype=torch.float32, device=device
        )
        self.u = torch.full(
            (num_neurons,), params.u_init, dtype=torch.float32, device=device
        )
        self.spike = torch.zeros(num_neurons, dtype=torch.float32, device=device)

        # Spike time tracking (for KC and EN, used in STDP)
        if track_spike_time:
            self.t_spike = torch.full(
                (num_neurons,), -10000.0, dtype=torch.float32, device=device
            )

    def reset(self):
        """Re-initialize all state variables (called at start of each image)."""
        self.v.fill_(self.params.v_init)
        self.u.fill_(self.params.u_init)
        self.spike.zero_()
        if self.track_spike_time:
            self.t_spike.fill_(-10000.0)

    def step(self, dt: float, t: float, I: torch.Tensor):
        """Advance one timestep.

        Args:
            dt: Time step (ms).
            t: Current time (ms). Only used if track_spike_time is True.
            I: Input current tensor (num_neurons,).
        """
        p = self.params

        # Threshold check and reset (matches MATLAB: if v > v_t)
        fired = self.v > p.v_t
        self.spike.zero_()
        self.spike[fired] = 1.0
        self.v[fired] = p.c
        self.u[fired] = self.u[fired] + p.d

        if self.track_spike_time:
            self.t_spike[fired] = t

        # Noise term — sampled once, reused in both half-steps
        epsilon = p.epsilon_mean + p.epsilon_std * torch.randn(
            self.num_neurons, dtype=torch.float32, device=self.device
        )

        # Half-step voltage update (applied twice, same epsilon)
        dv = (
            p.k * (self.v - p.v_r) * (self.v - p.v_t)
            - self.u
            + I
            + epsilon
        ) / p.C
        self.v = self.v + (dt / 2.0) * dv

        dv = (
            p.k * (self.v - p.v_r) * (self.v - p.v_t)
            - self.u
            + I
            + epsilon
        ) / p.C
        self.v = self.v + (dt / 2.0) * dv

        # Recovery variable update
        self.u = self.u + dt * p.a * (p.b * (self.v - p.v_r) - self.u)


class PNLayer(IzhikevichLayer):
    """Projection Neuron layer (360 neurons)."""

    def __init__(self, num_neurons: int = 360, params: PNParams = None,
                 device: torch.device = None):
        if params is None:
            params = PNParams()
        super().__init__(num_neurons, params, device, track_spike_time=False)


class KCLayer(IzhikevichLayer):
    """Kenyon Cell layer (20,000 neurons)."""

    def __init__(self, num_neurons: int = 20000, params: KCParams = None,
                 device: torch.device = None):
        if params is None:
            params = KCParams()
        super().__init__(num_neurons, params, device, track_spike_time=True)


class ENLayer(IzhikevichLayer):
    """Extrinsic Neuron layer (1 neuron)."""

    def __init__(self, num_neurons: int = 1, params: ENParams = None,
                 device: torch.device = None):
        if params is None:
            params = ENParams()
        super().__init__(num_neurons, params, device, track_spike_time=True)
