"""Deterministic PN/KC circuit; no optimizers, stochastic spike encoders or plasticity.

Voltages are expressed relative to rest, in units of the PN firing threshold.
The LIF time constant (10 ms) and refractory period (2 ms) follow Jesusanmi
et al. (2024), doi:10.3389/fphys.2024.1379977. Adaptation and finite graded
inhibition are modelling hypotheses, not fitted physiological measurements.
"""
from dataclasses import dataclass, asdict
import math

import torch
from torch import nn


@dataclass(frozen=True)
class CircuitConfig:
    duration_ms: float = 50.0
    dt_ms: float = 1.0
    tau_mem_ms: float = 10.0
    refractory_ms: float = 2.0
    tau_syn_ms: float = 3.0
    tau_feedback_ms: float = 5.0
    tau_adapt_ms: float = 50.0
    pn_gain: float = 3.0
    synaptic_gain: float = 8.0
    feedback_gain: float = 40.0
    adaptation_jump: float = 0.6
    threshold: float = 0.3

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("duration_ms", "dt_ms", "tau_mem_ms", "tau_syn_ms",
                     "tau_feedback_ms", "tau_adapt_ms", "threshold"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dt_ms > min(self.tau_mem_ms, self.tau_syn_ms, self.tau_feedback_ms):
            raise ValueError("dt_ms must not exceed the shortest circuit time constant")
        if not math.isclose(self.duration_ms / self.dt_ms, round(self.duration_ms / self.dt_ms)):
            raise ValueError("duration_ms must be a multiple of dt_ms")


class SparseCircuit(nn.Module):
    """Fixed excitatory projection plus per-view LIF dynamics.

    Connectivity uses a private CPU generator and is saved as an index buffer.
    All dynamic state is local to simulate(): batch members and calls cannot
    influence each other. A returned state can explicitly continue a sequence.
    """
    def __init__(self, pn_dim, code_dim=4000, fan_in=10, seed=7, config=None):
        super().__init__()
        if code_dim < 1 or not 1 <= fan_in <= pn_dim:
            raise ValueError("code_dim must be positive and fan_in in [1, pn_dim]")
        self.config = config or CircuitConfig()
        self.pn_dim, self.code_dim, self.fan_in = pn_dim, code_dim, fan_in
        gen = torch.Generator().manual_seed(seed)
        indices = torch.stack([torch.randperm(pn_dim, generator=gen)[:fan_in]
                               for _ in range(code_dim)])
        self.register_buffer("indices", indices)
        self.register_buffer("threshold", torch.tensor(self.config.threshold))

    def project(self, x):
        return x[:, self.indices].mean(-1)

    @torch.no_grad()
    def pn_spikes(self, inputs, state=None):
        c = self.config
        n, p = inputs.shape
        if p != self.pn_dim or not torch.isfinite(inputs).all() or (inputs < 0).any():
            raise ValueError("PN inputs must be finite, nonnegative [batch, pn_dim]")
        if state is not None and any(state[k].shape != inputs.shape for k in ("pn_v", "pn_ref")):
            raise ValueError("Continuation state must match the input batch and PN dimensions")
        v = inputs.new_zeros(n, p) if state is None else state["pn_v"].clone()
        ref = torch.zeros_like(v) if state is None else state["pn_ref"].clone()
        decay = math.exp(-c.dt_ms / c.tau_mem_ms)
        events = []
        for _ in range(round(c.duration_ms / c.dt_ms)):
            ready = ref <= 0
            v = torch.where(ready, decay * v + (1 - decay) * c.pn_gain * inputs, 0.)
            fired = v >= 1.0
            v.masked_fill_(fired, 0.)
            ref = torch.where(fired, c.refractory_ms, (ref - c.dt_ms).clamp_min(0))
            events.append(fired)
        return torch.stack(events), {"pn_v": v, "pn_ref": ref}

    @torch.no_grad()
    def simulate(self, inputs, mode="adaptive", readout="binary", perturbation="none",
                 record=False, state=None):
        if mode not in ("threshold", "feedback", "adaptive"):
            raise ValueError(f"Unknown spiking mode: {mode}")
        if readout not in ("binary", "count", "early"):
            raise ValueError(f"Unknown spike readout: {readout}")
        if perturbation not in ("none", "no-inhibition", "no-adaptation", "shuffle-times"):
            raise ValueError(f"Unknown perturbation: {perturbation}")
        c = self.config
        pn, next_state = self.pn_spikes(inputs, state)
        if perturbation == "shuffle-times":
            # Different fixed permutation for every PN, shared across batch members.
            # Counts are exactly preserved, including for silent PNs.
            gen = torch.Generator().manual_seed(1729)
            perm = torch.stack([torch.randperm(len(pn), generator=gen)
                                for _ in range(self.pn_dim)], dim=1).to(pn.device)
            pn = pn.gather(0, perm[:, None, :].expand_as(pn))
        n = len(inputs)
        def initial(key, width):
            if state is not None and state[key].shape != (n, width):
                raise ValueError(f"Continuation state shape mismatch: {key}")
            return inputs.new_zeros(n, width) if state is None else state[key].clone()
        v, ref = initial("kc_v", self.code_dim), initial("kc_ref", self.code_dim)
        adapt = initial("adapt", self.code_dim)
        syn = initial("syn", self.code_dim)
        feedback = initial("feedback", 1)
        counts = torch.zeros_like(v)
        early = torch.zeros_like(v)
        first = torch.full_like(v, float("inf"))
        rasters, inhibitory_trace = [], []
        dm, ds, df, da = [math.exp(-c.dt_ms / tau) for tau in
                         (c.tau_mem_ms, c.tau_syn_ms, c.tau_feedback_ms, c.tau_adapt_ms)]
        inhibition_on = mode != "threshold" and perturbation != "no-inhibition"
        adaptation_on = mode == "adaptive" and perturbation != "no-adaptation"
        for t, spikes in enumerate(pn):
            # PSC impulse amplitude independent of dt; integral converges as dt shrinks.
            syn = ds * syn + c.synaptic_gain * self.project(spikes.to(inputs.dtype))
            adapt = da * adapt
            feedback = df * feedback
            current = syn - (feedback if inhibition_on else 0) - (adapt if adaptation_on else 0)
            v = torch.where(ref <= 0, dm * v + (1 - dm) * current, 0.)
            fired = v >= self.threshold
            v.masked_fill_(fired, 0.)
            ref = torch.where(fired, c.refractory_ms, (ref - c.dt_ms).clamp_min(0))
            if adaptation_on:
                adapt = adapt + c.adaptation_jump * fired
            if inhibition_on:
                feedback = feedback + c.feedback_gain * fired.float().mean(1, keepdim=True)
            first = torch.where(fired & torch.isinf(first), (t + 1) * c.dt_ms, first)
            counts += fired
            if (t + 1) * c.dt_ms <= min(20., c.duration_ms):
                early += fired
            if record:
                rasters.append(fired)
                inhibitory_trace.append(feedback.clone())
        next_state.update(kc_v=v, kc_ref=ref, syn=syn, adapt=adapt, feedback=feedback)
        codes = {"binary": (counts > 0).float(), "count": counts, "early": (early > 0).float()}[readout]
        result = {"codes": codes, "counts": counts, "first_spike_ms": first,
                  "pn_counts": pn.sum(0), "state": next_state}
        if record:
            result.update(pn_spikes=pn, kc_spikes=torch.stack(rasters),
                          inhibition=torch.stack(inhibitory_trace))
        return result
