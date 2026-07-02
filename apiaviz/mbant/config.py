"""Default parameters matching the MATLAB implementation exactly."""

from dataclasses import dataclass


@dataclass
class PNParams:
    """Izhikevich parameters for Projection Neurons (PN_neuron.m)."""

    C: float = 100.0
    a: float = 0.3
    b: float = -0.2
    c: float = -65.0
    d: float = 8.0
    k: float = 2.0
    v_r: float = -60.0
    v_t: float = -40.0
    epsilon_mean: float = 0.0
    epsilon_std: float = 0.05
    v_init: float = -60.0
    u_init: float = 0.0


@dataclass
class KCParams:
    """Izhikevich parameters for Kenyon Cells (KC_neuron.m)."""

    C: float = 4.0
    a: float = 0.01
    b: float = -0.3
    c: float = -65.0
    d: float = 8.0
    k: float = 0.035
    v_r: float = -85.0
    v_t: float = -25.0
    epsilon_mean: float = 0.0
    epsilon_std: float = 0.05
    v_init: float = -85.0
    u_init: float = 0.0


@dataclass
class ENParams:
    """Izhikevich parameters for Extrinsic Neuron (EN_neuron.m)."""

    C: float = 100.0
    a: float = 0.3
    b: float = -0.2
    c: float = -65.0
    d: float = 8.0
    k: float = 2.0
    v_r: float = -60.0
    v_t: float = -40.0
    epsilon_mean: float = 0.0
    epsilon_std: float = 0.05
    v_init: float = -60.0
    u_init: float = 0.0


@dataclass
class PNKCSynapseParams:
    """Parameters for PN->KC synapse (PN_KC_synapse.m)."""

    phi_S: float = 0.93
    tau_syn_S: float = 3.0


@dataclass
class KCENSynapseParams:
    """Parameters for KC->EN synapse (KC_EN_synapse.m)."""

    phi_S: float = 8.0
    tau_syn_S: float = 8.0
    tau_c: float = 40.0
    tau_d: float = 20.0


@dataclass
class STDPParams:
    """Parameters for STDP rule (STDP.m)."""

    A_plus: float = 1.0
    A_minus: float = -1.0
    tau_plus: float = 15.0
    tau_minus: float = 15.0


@dataclass
class NetworkConfig:
    """Full network configuration (ant_route_following_test.m)."""

    # Architecture
    numPN: int = 360
    numKC: int = 20000
    numEN: int = 1
    num_pn_per_kc: int = 10

    # Synaptic conductances
    g_PN_KC: float = 0.25
    g_KC_EN: float = 2.0

    # Input scaling
    C_I_PN_var: float = 5250.0

    # Simulation
    interval: int = 50
    dt: float = 1.0

    # Reward
    ba_magnitude: float = 0.5
    ba_release_time: float = 40.0

    # Neuron parameters
    pn_params: PNParams = None
    kc_params: KCParams = None
    en_params: ENParams = None
    pn_kc_synapse_params: PNKCSynapseParams = None
    kc_en_synapse_params: KCENSynapseParams = None
    stdp_params: STDPParams = None

    def __post_init__(self):
        if self.pn_params is None:
            self.pn_params = PNParams()
        if self.kc_params is None:
            self.kc_params = KCParams()
        if self.en_params is None:
            self.en_params = ENParams()
        if self.pn_kc_synapse_params is None:
            self.pn_kc_synapse_params = PNKCSynapseParams()
        if self.kc_en_synapse_params is None:
            self.kc_en_synapse_params = KCENSynapseParams()
        if self.stdp_params is None:
            self.stdp_params = STDPParams()


@dataclass
class NavigationConfig:
    """Navigation/route following parameters."""

    step_size: float = 0.1  # [m]
    scan_range: float = 120.0  # [degrees] total scan range
    scan_step: float = 2.0  # [degrees] angular resolution
    dis_threshold: float = 0.2  # [m] off-route error threshold
    eye_height: float = 0.01  # [m]
    resolution: float = 4.0  # [degrees/pixel]
    hfov: float = 296.0  # [degrees]

    @property
    def num_scan_img(self) -> int:
        """Number of headings to scan (120/2 + 1 = 61)."""
        return int(self.scan_range / self.scan_step) + 1


@dataclass
class ImageConfig:
    """Image capture and preprocessing parameters."""

    eye_height: float = 0.01  # [m]
    resolution: float = 4.0  # [degrees/pixel]
    hfov: float = 296.0  # [degrees]
    img_separation: int = 10  # [cm] between training images
    resize_shape: tuple = (10, 36)  # (height, width) after resize
