# Imports
import torch, math

import torch.nn as nn
import torch.nn.functional as F

from .functional import APLCompetition, SparseLinear

class VisionBackbone(nn.Module):
    def __init__(self, lobula_dim=32, lobula_embedding_dim=128, lobula_plate_dim=32):
        super().__init__()

        # TODO: Modify the retina input to handle the BG and greyscale as two separate retina layers
        # TODO: Match channels to ommatidial count, lamina cartridge, and medulla columns

        # ───── Ommatidial ─────
        # Two separate inputs - R1-R6 carrying achromatic information to the lamina
        # R8 carrying color information directly to the medulla
        self.R1_R6 = HexRouting2d(1, learnable=True, bias=True) # 1 input channel per color, learnable hex routing
        self.R8 = R8() # Minimal R8-like chromatic encoder, 2 input channels (G, B), 1 output channel

        # ───── Lamina ─────
        # Lamina receives R1_R6 hex-routed input, with three channels to represent L1-L3 features
        self.lamina = Lamina()

        # ───── Medulla: Color & Achromatic Pathways ─────
        self.medulla = Medulla() # Non-generic medulla with M1, M2, M3 pathways; M3 integrates L3 and R8

        # ───── Lobula (higher-order feature integration) ─────
        self.lobula = Lobula(feature_channels=lobula_dim, embedding_dim=lobula_embedding_dim) # Features
        self.lobula_plate = LobulaPlate(in_channels=lobula_dim, out_channels=lobula_plate_dim) # Spatial features

    def _split_input(self, x):
        if x.size(1) == 2:
            achromatic = x.mean(dim=1, keepdim=True)
            chromatic = x
        elif x.size(1) >= 3:
            achromatic = x.mean(dim=1, keepdim=True)
            chromatic = x[:, 1:3]
        else:
            raise ValueError(f"VisionBackbone expected 2 or 3 input channels, got {x.size(1)}")
        return achromatic, chromatic

    def forward(self, x, detach_lobula_for_plate=False, return_maps=False):
        achromatic, chromatic = self._split_input(x)

        # Ommatidial activation
        lam_in = self.R1_R6(achromatic)
        r8_map = self.R8(chromatic)

        # Lamina: apply spatial filtering
        lam_out = self.lamina(lam_in)

        # Medulla: combine chromatic and achromatic processing
        med = self.medulla(lam_out, r8_map)

        # Lobula: integrate complex features
        lob = self.lobula(med["M1"], med["M2"], med["M3"])
        lobula_features = lob["feature_map"].detach() if detach_lobula_for_plate else lob["feature_map"]
        lobula_plate = self.lobula_plate(lobula_features)

        if return_maps:
            return {
                "achromatic": achromatic,
                "lam_in": lam_in,
                "lam_out": lam_out,
                "r8_map": r8_map,
                **med,
                "lobula": lob["embedding"],
                "lobula_feature_map": lob["feature_map"],
                "lobula_gem_p": lob["gem_p"],
                "lobula_plate": lobula_plate,
                "medulla_mix": lob["medulla_mix"],
            }

        return lob["embedding"], lobula_plate

class HexRouting2d(nn.Module):
    def __init__(self, in_channels, out_channels=1, learnable=False, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if learnable:
            self.weight = nn.Parameter(torch.ones(out_channels, in_channels, 6) / 6.0)
        else:
            w = torch.ones(out_channels, in_channels, 6, dtype=torch.float32) / 6.0
            self.register_buffer("weight", w)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        B, C, H, W = x.shape
        xp = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="reflect")

        left  = xp[:, :, 1:H+1, 0:W]
        right = xp[:, :, 1:H+1, 2:W+2]

        up_left    = xp[:, :, 0:H,   0:W]
        up_mid     = xp[:, :, 0:H,   1:W+1]
        up_right   = xp[:, :, 0:H,   2:W+2]

        down_left  = xp[:, :, 2:H+2, 0:W]
        down_mid   = xp[:, :, 2:H+2, 1:W+1]
        down_right = xp[:, :, 2:H+2, 2:W+2]

        row_idx = torch.arange(H, device=x.device).view(1, 1, H, 1)
        even_mask = (row_idx % 2 == 0).to(x.dtype)
        odd_mask = 1.0 - even_mask

        # odd-r hex layout
        up_a   = even_mask * up_mid     + odd_mask * up_left
        up_b   = even_mask * up_right   + odd_mask * up_mid
        down_a = even_mask * down_mid   + odd_mask * down_left
        down_b = even_mask * down_right + odd_mask * down_mid

        neigh = torch.stack([left, right, up_a, up_b, down_a, down_b], dim=2)
        y = torch.einsum("ocn,bcnhw->bohw", self.weight, neigh)

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)

        return y
    
class R8(nn.Module):
    """
    Minimal R8-like chromatic encoder.

    Input:
        chromatic: [B, 2, H, W]   e.g. [G, B]

    Output:
        r8:        [B, 1, H, W]
    """
    def __init__(self):
        super().__init__()
        self.spectral = nn.Conv2d(2, 1, kernel_size=1, bias=True)

        with torch.no_grad():
            self.spectral.weight.zero_()
            self.spectral.bias.zero_()
            self.spectral.weight[0, 0, 0, 0] = 0.5
            self.spectral.weight[0, 1, 0, 0] = 0.5

    def forward(self, chromatic):
        return F.relu(self.spectral(chromatic))
    
class Lamina(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        assert kernel_size % 2 == 1

        self.filters = nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="reflect",
            bias=True
        )

        self._init_filters(kernel_size)

    def _init_filters(self, k):
        lp = torch.ones(k, k) / (k * k)

        # ON-like: positive center, negative surround
        on = -lp.clone()
        on[k // 2, k // 2] += 1.0

        # OFF-like: negative center, positive surround
        off = lp.clone()
        off[k // 2, k // 2] -= 1.0

        with torch.no_grad():
            self.filters.weight.zero_()
            self.filters.bias.zero_()

            self.filters.weight[0, 0] = on   # L1
            self.filters.weight[1, 0] = off  # L2
            self.filters.weight[2, 0] = lp   # L3

    def forward(self, x):
        y = self.filters(x)

        L1 = F.relu(y[:, 0:1])   # ON-like
        L2 = F.relu(y[:, 1:2])   # OFF-like
        L3 = y[:, 2:3]           # sustained / luminance

        return torch.cat([L1, L2, L3], dim=1)

class M1Pathway(nn.Module):
    """
    M1-like medulla node driven mainly by L1.
    """
    def __init__(self):
        super().__init__()
        self.gain = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            self.gain.weight.fill_(1.0)
            self.gain.bias.zero_()

    def forward(self, L1):
        return F.relu(self.gain(L1))


class M2Pathway(nn.Module):
    """
    M2-like medulla node driven mainly by L2.
    """
    def __init__(self):
        super().__init__()
        self.gain = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            self.gain.weight.fill_(1.0)
            self.gain.bias.zero_()

    def forward(self, L2):
        return F.relu(self.gain(L2))


class M3Pathway(nn.Module):
    """
    Dedicated M3 integration of:
        - L3 achromatic/luminance signal
        - R8 chromatic signal

    """
    def __init__(self):
        super().__init__()
        self.integrate = nn.Conv2d(2, 1, kernel_size=1, bias=True)
        self.local = nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode="reflect", bias=True)

        with torch.no_grad():
            self.integrate.weight.zero_()
            self.integrate.bias.zero_()
            self.integrate.weight[0, 0, 0, 0] = 0.5   # L3
            self.integrate.weight[0, 1, 0, 0] = 0.5   # R8

            self.local.weight.zero_()
            self.local.bias.zero_()
            self.local.weight[0, 0] = torch.ones(3, 3) / 9.0

    def forward(self, L3, R8):
        x = torch.cat([L3, R8], dim=1)
        x = self.integrate(x)
        x = self.local(x)
        return F.relu(x)


class Medulla(nn.Module):
    """
    Non-generic medulla:
        M1 <- L1
        M2 <- L2
        M3 <- [L3, R8]

    """
    def __init__(self):
        super().__init__()
        self.M1 = M1Pathway()
        self.M2 = M2Pathway()
        self.M3 = M3Pathway()

    def forward(self, lamina_out, r8_map):
        L1 = lamina_out[:, 0:1]
        L2 = lamina_out[:, 1:2]
        L3 = lamina_out[:, 2:3]

        M1 = self.M1(L1)
        M2 = self.M2(L2)
        M3 = self.M3(L3, r8_map)

        return {
            "M1": M1,
            "M2": M2,
            "M3": M3,
        }

class MedullaFeatureMixer(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, pool_scales=(1, 2, 4)):
        super().__init__()
        self.pool_scales = tuple(pool_scales)
        self.local = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect", bias=True)
        self.context = nn.Conv2d(in_channels * len(self.pool_scales), out_channels, kernel_size=1, bias=True)
        self.refine = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, padding_mode="reflect", bias=True)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)

    def _pool_context(self, x):
        pooled = []
        for scale in self.pool_scales:
            if scale == 1:
                pooled.append(x)
                continue
            p = F.avg_pool2d(x, kernel_size=scale, stride=scale, ceil_mode=True)
            p = F.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)
            pooled.append(p)
        return torch.cat(pooled, dim=1)

    def forward(self, x):
        local = self.local(x)
        context = self.context(self._pool_context(x))
        fused = torch.cat([local, context], dim=1)
        return F.relu(self.norm(self.refine(fused)))

class GeneralizedMeanPooling2d(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=self.eps)
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / p)

class Lobula(nn.Module):
    def __init__(self, medulla_channels=3, feature_channels=32, embedding_dim=128):
        super().__init__()
        self.mixer = MedullaFeatureMixer(in_channels=medulla_channels, out_channels=feature_channels)
        self.integrate = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, padding_mode="reflect", bias=True)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=feature_channels)
        self.gem_pool = GeneralizedMeanPooling2d(p=3.0)
        self.embedding = nn.Linear(feature_channels, embedding_dim)

    def forward(self, M1, M2, M3):
        medulla_features = torch.cat([M1, M2, M3], dim=1)
        mixed = self.mixer(medulla_features)
        feature_map = F.relu(self.norm(self.integrate(mixed) + mixed))
        pooled = self.gem_pool(feature_map).flatten(1)
        embedding = self.embedding(pooled)

        return {
            "feature_map": feature_map,
            "embedding": embedding,
            "medulla_mix": mixed,
            "gem_p": self.gem_pool.p.clamp(min=self.gem_pool.eps).detach(),
        }

class LobulaPlate(nn.Module):
    def __init__(self, in_channels=32, out_channels=32):
        super().__init__()
        self.project = nn.Conv2d(in_channels + 2, out_channels, kernel_size=1, bias=True)
        self.local = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect", groups=out_channels, bias=True)
        self.context = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=True)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)

    def _coords(self, x):
        _, _, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype),
            indexing="ij"
        )
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0)
        return coords.expand(x.size(0), -1, -1, -1)

    def forward(self, lobula_features):
        x = torch.cat([lobula_features, self._coords(lobula_features)], dim=1)
        x = F.relu(self.project(x))
        local = self.local(x)
        pooled = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
        dense_map = self.context(torch.cat([local, pooled], dim=1))
        return F.relu(self.norm(dense_map + x))

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(out_dim * 2, min(in_dim, 512))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class CrossAttentionPool(nn.Module):
    def __init__(self, token_dim: int, num_slots: int):
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_slots = int(num_slots)
        self.query = nn.Parameter(torch.randn(self.num_slots, self.token_dim) / math.sqrt(self.token_dim))
        self.norm = nn.LayerNorm(self.token_dim)

    def forward(self, tokens: torch.Tensor):
        """
        tokens: [B, N, D]
        returns:
            slots: [B, S, D]
            attn:  [B, S, N]
        """
        if tokens.ndim != 3:
            raise ValueError(f"CrossAttentionPool expected [B, N, D], got {tuple(tokens.shape)}")

        bsz, _, dim = tokens.shape
        query = self.query.unsqueeze(0).expand(bsz, -1, -1)  # [B, S, D]
        attn = torch.softmax(torch.matmul(query, tokens.transpose(1, 2)) / math.sqrt(dim), dim=-1)
        slots = torch.matmul(attn, tokens)
        return self.norm(slots), attn


class FourierCoordEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(10, out_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(out_dim),
        )

    def forward(self, coords: torch.Tensor):
        """
        coords: [B, N, 2] in [-1, 1]
        """
        x = coords[..., 0:1]
        y = coords[..., 1:2]
        feats = torch.cat(
            [
                x,
                y,
                x * y,
                x.square(),
                y.square(),
                torch.sin(math.pi * x),
                torch.cos(math.pi * x),
                torch.sin(math.pi * y),
                torch.cos(math.pi * y),
                torch.sin(math.pi * (x + y)),
            ],
            dim=-1,
        )
        return self.project(feats)


class FeatureVPN(nn.Module):
    """
    Content-focused vPN population.

    Goal:
    - stable object/content representation
    - moderate translation tolerance
    - avoids collapsing everything into one global descriptor too early
    """
    def __init__(self, embedding_dim=128, feature_channels=32, out_dim=128):
        super().__init__()
        self.pool_size = 4
        self.token_dim = max(32, out_dim // 2)
        self.num_slots = 4

        self.feature_pool = GeneralizedMeanPooling2d(p=3.0)

        self.local_reduce = nn.Conv2d(feature_channels, self.token_dim, kernel_size=1, bias=True)
        self.local_refine = nn.Conv2d(
            self.token_dim,
            self.token_dim,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
            bias=True,
        )
        self.local_norm = nn.GroupNorm(num_groups=1, num_channels=self.token_dim)

        self.slot_pool = CrossAttentionPool(token_dim=self.token_dim, num_slots=self.num_slots)
        self.project = ProjectionMLP(
            embedding_dim + feature_channels + (2 * self.token_dim),
            out_dim,
        )

    def forward(self, lobula_embedding, lobula_feature_map):
        pooled_feature_map = self.feature_pool(lobula_feature_map).flatten(1)

        local = F.relu(self.local_reduce(lobula_feature_map))
        local = F.relu(self.local_norm(self.local_refine(local)))

        token_map = F.adaptive_avg_pool2d(local, self.pool_size)
        tokens = token_map.flatten(2).transpose(1, 2)  # [B, N, D]

        slots, _ = self.slot_pool(tokens)
        slot_mean = slots.mean(dim=1)
        slot_max = slots.amax(dim=1)
        slot_summary = torch.cat([slot_mean, slot_max], dim=1)

        descriptor = self.project(
            torch.cat([lobula_embedding, pooled_feature_map, slot_summary], dim=1)
        )

        return {
            "descriptor": descriptor,
            "pooled_feature_map": pooled_feature_map,
        }
    
class SpatialVPN(nn.Module):
    """
    Spatial/context vPN population.

    Goal:
    - keep coarse position and layout
    - be less brittle than raw retinotopy
    - expose position-aware tokens to the conjunctive pathway
    """
    def __init__(self, in_channels=32, pool_size=4, token_dim=64, out_dim=128):
        super().__init__()
        self.pool_size = int(pool_size)
        self.token_dim = int(token_dim)
        self.coord_dim = max(8, self.token_dim // 4)

        self.pre = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_groups=1, num_channels=in_channels),
        )

        self.coord_encoder = FourierCoordEncoder(out_dim=self.coord_dim)
        self.token_proj = nn.Linear(in_channels + self.coord_dim, self.token_dim)
        self.token_refine = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.token_score = nn.Linear(self.token_dim, 1)

        self.project = ProjectionMLP(2 * self.token_dim, out_dim)

    def _coords(self, x):
        _, _, height, width = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0)
        return coords.expand(x.size(0), -1, -1, -1)

    def forward(self, lobula_plate):
        dense = self.pre(lobula_plate)
        pooled_map = F.adaptive_avg_pool2d(dense, self.pool_size)

        coords = self._coords(pooled_map).flatten(2).transpose(1, 2)  # [B, N, 2]
        pooled_tokens = pooled_map.flatten(2).transpose(1, 2)         # [B, N, C]
        coord_tokens = self.coord_encoder(coords)                     # [B, N, coord_dim]

        token_features = self.token_proj(torch.cat([pooled_tokens, coord_tokens], dim=-1))
        token_features = token_features + self.token_refine(token_features)

        token_weights = torch.softmax(self.token_score(torch.tanh(token_features)).squeeze(-1), dim=1)
        pooled_tokens_summary = torch.sum(token_features * token_weights.unsqueeze(-1), dim=1)

        token_spread = torch.mean(
            torch.abs(token_features - pooled_tokens_summary.unsqueeze(1)),
            dim=1,
        )

        descriptor = self.project(torch.cat([pooled_tokens_summary, token_spread], dim=1))

        return {
            "descriptor": descriptor,
            "tokens": token_features,
            "token_weights": token_weights,
            "pooled_tokens": pooled_tokens_summary,
            "pooled_map": pooled_map,
        }


class ConjunctiveVPN(nn.Module):
    """
    Mixed-selectivity vPN population.

    Goal:
    - bind stable content with spatial context
    - same object in a new place should not produce the same code
    - small changes preserve partial overlap rather than total identity
    """
    def __init__(self, feature_dim=128, spatial_dim=128, token_dim=64, out_dim=128):
        super().__init__()
        self.feature_gate = nn.Linear(feature_dim, token_dim)
        self.feature_bias = nn.Linear(feature_dim, token_dim)
        self.spatial_context = nn.Linear(spatial_dim, token_dim)

        self.bound_refine = nn.Sequential(
            nn.LayerNorm(3 * token_dim),
            nn.Linear(3 * token_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim),
        )
        self.token_score = nn.Linear(token_dim, 1)
        self.project = ProjectionMLP(feature_dim + spatial_dim + token_dim, out_dim)

    def forward(self, feature_descriptor, spatial_descriptor, spatial_tokens):
        gate = torch.sigmoid(self.feature_gate(feature_descriptor)).unsqueeze(1)         # [B, 1, D]
        bias = self.feature_bias(feature_descriptor).unsqueeze(1)                        # [B, 1, D]
        spatial_ctx = torch.tanh(self.spatial_context(spatial_descriptor)).unsqueeze(1)  # [B, 1, D]

        gated = spatial_tokens * gate
        interaction = gated * spatial_ctx

        bound_tokens = self.bound_refine(
            torch.cat(
                [
                    spatial_tokens,
                    gated + bias,
                    interaction,
                ],
                dim=-1,
            )
        )

        token_weights = torch.softmax(self.token_score(torch.tanh(bound_tokens)).squeeze(-1), dim=1)
        bound_summary = torch.sum(bound_tokens * token_weights.unsqueeze(-1), dim=1)

        descriptor = self.project(
            torch.cat([feature_descriptor, spatial_descriptor, bound_summary], dim=1)
        )

        return {
            "descriptor": descriptor,
            "bound_tokens": bound_tokens,
            "token_weights": token_weights,
            "bound_summary": bound_summary,
            "gate": gate.squeeze(1),
        }
    
class VisionProjection(nn.Module):
    def __init__(
        self,
        lobula_dim=128,
        lobula_feature_channels=32,
        lobula_plate_channels=32,
        vpn_dim=128,
        spatial_pool_size=4,
        spatial_token_dim=64,
        kc_dim=2048,
        kc_fan_in=8,
        kc_sparsity=0.03,
        apl_feedback_strength=0.05,
        apl_gain_adapt_rate=0.25,
        apl_threshold_lr=0.02,
        apl_num_iters=3,
    ):
        super().__init__()
        self.feature_vpn = FeatureVPN(
            embedding_dim=lobula_dim,
            feature_channels=lobula_feature_channels,
            out_dim=vpn_dim,
        )
        self.spatial_vpn = SpatialVPN(
            in_channels=lobula_plate_channels,
            pool_size=spatial_pool_size,
            token_dim=spatial_token_dim,
            out_dim=vpn_dim,
        )
        self.conjunctive_vpn = ConjunctiveVPN(
            feature_dim=vpn_dim,
            spatial_dim=vpn_dim,
            token_dim=spatial_token_dim,
            out_dim=vpn_dim,
        )

        feature_fan_in, spatial_fan_in, conjunctive_fan_in = self._split_kc_fan_in(kc_fan_in)

        self.kc_projection = SparseLinear(
            3 * vpn_dim,
            kc_dim,
            fan_in=kc_fan_in,
            bias=False,
            group_slices=[
                (0, vpn_dim),
                (vpn_dim, 2 * vpn_dim),
                (2 * vpn_dim, 3 * vpn_dim),
            ],
            group_fan_in=[
                feature_fan_in,
                spatial_fan_in,
                conjunctive_fan_in,
            ],
        )
        self.kc_compete = APLCompetition(
            num_units=kc_dim,
            target_sparsity=kc_sparsity,
            feedback_strength=apl_feedback_strength,
            gain_adapt_rate=apl_gain_adapt_rate,
            threshold_lr=apl_threshold_lr,
            num_iters=apl_num_iters,
        )

    @staticmethod
    def _split_kc_fan_in(kc_fan_in: int):
        kc_fan_in = int(kc_fan_in)
        if kc_fan_in < 3:
            return 1, 1, max(1, kc_fan_in - 2)

        feature_fan_in = max(1, kc_fan_in // 4)
        spatial_fan_in = max(1, kc_fan_in // 4)
        conjunctive_fan_in = kc_fan_in - feature_fan_in - spatial_fan_in

        if conjunctive_fan_in < 1:
            conjunctive_fan_in = 1
            if spatial_fan_in > feature_fan_in:
                spatial_fan_in -= 1
            else:
                feature_fan_in -= 1

        return feature_fan_in, spatial_fan_in, conjunctive_fan_in

    def _unpack_inputs(self, lobula, lobula_feature_map=None, lobula_plate=None):
        if isinstance(lobula, dict):
            outputs = lobula
            lobula_feature_map = outputs.get("lobula_feature_map")
            lobula_plate = outputs.get("lobula_plate")
            lobula = outputs.get("lobula")

        if lobula is None or lobula_feature_map is None or lobula_plate is None:
            raise ValueError(
                "VisionProjection expects lobula embedding, lobula feature map, and lobula plate map. "
                "Pass either three tensors or the return_maps dictionary from VisionBackbone."
            )

        return lobula, lobula_feature_map, lobula_plate

    def forward(self, lobula, lobula_feature_map=None, lobula_plate=None):
        lobula, lobula_feature_map, lobula_plate = self._unpack_inputs(
            lobula,
            lobula_feature_map=lobula_feature_map,
            lobula_plate=lobula_plate,
        )

        feature_outputs = self.feature_vpn(lobula, lobula_feature_map)
        spatial_outputs = self.spatial_vpn(lobula_plate)
        conjunctive_outputs = self.conjunctive_vpn(
            feature_outputs["descriptor"],
            spatial_outputs["descriptor"],
            spatial_outputs["tokens"],
        )

        vpn = torch.cat(
            [
                feature_outputs["descriptor"],
                spatial_outputs["descriptor"],
                conjunctive_outputs["descriptor"],
            ],
            dim=1,
        )

        kenyon_drive = F.relu(self.kc_projection(vpn))
        kenyon_code = self.kc_compete(kenyon_drive)
        kc_active_counts = (kenyon_code > 0).sum(dim=1)
        kc_active_fraction = (kenyon_code > 0).float().mean(dim=1)

        return {
            "feature_vpn": feature_outputs["descriptor"],
            "feature_pool": feature_outputs["pooled_feature_map"],
            "spatial_vpn": spatial_outputs["descriptor"],
            "spatial_tokens": spatial_outputs["tokens"],
            "spatial_token_weights": spatial_outputs["token_weights"],
            "spatial_pooled_tokens": spatial_outputs["pooled_tokens"],
            "spatial_pooled_map": spatial_outputs["pooled_map"],
            "conjunctive_vpn": conjunctive_outputs["descriptor"],
            "conjunctive_tokens": conjunctive_outputs["bound_tokens"],
            "conjunctive_token_weights": conjunctive_outputs["token_weights"],
            "conjunctive_gate": conjunctive_outputs["gate"],
            "conjunctive_summary": conjunctive_outputs["bound_summary"],
            "vpn": vpn,
            "kenyon_drive": kenyon_drive,
            "kenyon_code": kenyon_code,
            "kc_active_counts": kc_active_counts,
            "kc_active_fraction": kc_active_fraction,
        }

class RewardMemoryHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(hidden_dim)
        dropout = float(dropout)

        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )
            readout_dim = hidden_dim
        else:
            self.net = nn.Identity()
            readout_dim = int(in_dim)

        self.readout = nn.Linear(readout_dim, 1)

    def forward(self, x):
        hidden = self.net(x)
        reward_logit = self.readout(hidden).squeeze(-1)
        reward_probability = torch.sigmoid(reward_logit)
        return {
            "reward_logit": reward_logit,
            "reward_probability": reward_probability,
        }


def resolve_kc_sparsity_target(
    kc_dim: int,
    kc_sparsity: float = 0.03,
    kc_target_active: int = 0,
):
    kc_dim = int(kc_dim)
    if kc_dim <= 0:
        raise ValueError(f"kc_dim must be positive, got {kc_dim}")

    target_active = int(kc_target_active)
    if target_active > 0:
        if target_active > kc_dim:
            raise ValueError(
                f"kc_target_active ({target_active}) cannot exceed kc_dim ({kc_dim})"
            )
        effective_sparsity = max(1.0 / float(kc_dim), float(target_active) / float(kc_dim))
        return float(effective_sparsity), int(max(1, target_active))

    kc_sparsity = float(kc_sparsity)
    if not 0.0 < kc_sparsity <= 1.0:
        raise ValueError(f"kc_sparsity must be in (0, 1], got {kc_sparsity}")

    effective_sparsity = max(1.0 / float(kc_dim), kc_sparsity)
    target_active = max(1, int(effective_sparsity * kc_dim))
    return float(effective_sparsity), int(target_active)
