# Imports
import torch

import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F

from .functional import AdaptiveKWTA, SparseLinear

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
        xp = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="constant", value=0.0)

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
        self.local = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)

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
        self.local = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.context = nn.Conv2d(in_channels * len(self.pool_scales), out_channels, kernel_size=1, bias=True)
        self.refine = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=True)
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
        self.integrate = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=True)
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
        self.local = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=True)
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

class VisionProjection(nn.Module):
    def __init__(self, vpn_ch=64, kc_dim=1024):
        super().__init__()
        # ───── VPN layers: distinct feature projections ─────
        # These correspond to three pathways:
        #   ASOT = anterior superior optic tract
        #   AIOT = anterior inferior optic tract
        #   LOT  = lateral optic tract
        self.asot = nn.Conv2d(48, vpn_ch, 1)
        self.aiot = nn.Conv2d(48, vpn_ch, 1)
        self.lot  = nn.Conv2d(32, vpn_ch, 1)

        # ───── Mushroom Body (Kenyon Cell projection) ─────
        # Sparse, high-dimensional representation using learned sparse weights
        self.kc_p = SparseLinear(3 * vpn_ch, kc_dim)

    def forward(self, lob):
        # Extract three different VPN pathway features
        vpn = torch.cat([
            self._gp(self.asot(lob[:, :48])),
            self._gp(self.aiot(lob[:, 48:96])),
            self._gp(self.lot(lob[:, 96:]))
        ], dim=1)
        kc_raw = self.kc_p(vpn)
        
        # Apply adaptive sparsity
        return self.sparsity(kc_raw)
