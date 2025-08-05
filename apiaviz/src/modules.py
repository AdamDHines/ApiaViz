# Imports
import torch

import torch.nn as nn
import torch.nn.functional as F

from .functional import AdaptiveKWTA, k_wta, SparseLinear

class VisionModule(nn.Module):
    def __init__(self, kc_dim=1024, lam_ch=12, vpn_ch=64, use_adaptive_kwta=True, training=False):
        super().__init__()
        self.training = training
        # ───── Retina to Photoreceptor (Opsin response) ─────
        # Two input channels (green, blue), processed independently into 6 channels
        self.opsin = nn.Conv2d(2, 6, 1, groups=2, bias=True)

        # ───── Lamina (early local motion + contrast detection) ─────
        # Depthwise convolution – each lamina channel processes its own input
        self.lamina = nn.Conv2d(lam_ch, lam_ch, 3, padding=1, padding_mode="reflect", groups=lam_ch, bias=True)

        # ───── Medulla: Color & Achromatic Pathways ─────
        # Chromatic pathway (grouped for green/blue separation)
        self.med_c = nn.Conv2d(lam_ch, 2 * lam_ch, 3, padding=1, padding_mode="reflect", groups=2)
        # Achromatic pathway (e.g. luminance-based edge detection)
        self.med_a = nn.Conv2d(lam_ch, 2 * lam_ch, 3, padding=1, padding_mode="reflect")
        # Normalization over feature groups (approximates lateral inhibition)
        self.med_n = nn.GroupNorm(12, 60)

        # ───── Lobula (higher-order feature integration) ─────
        self.lobula = nn.Sequential(
            nn.Conv2d(60, 128, 5, padding=2, padding_mode="reflect"),
            nn.ReLU()
        )

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

        # LayerNorm for lobula
        self.lobula_norm = nn.GroupNorm(num_groups=1, num_channels=128)
        self.lamina_norm = nn.GroupNorm(num_groups=1, num_channels=lam_ch)

        # Choose sparsity mechanism
        if use_adaptive_kwta:
            self.sparsity = AdaptiveKWTA(sparsity=0.05)
        else:
            self.sparsity = lambda x: k_wta(x, pct=0.05)

    def _initialize_weights(self):
        """Initialize weights to prevent dead neurons."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use gain-adjusted initialization for LeakyReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0.1)
                if m.bias is not None:
                    # Small positive bias to help with dead neurons
                    nn.init.constant_(m.bias, 0.01)

    def _gp(self, f):
        # Global average pooling over spatial dimensions
        return F.adaptive_avg_pool2d(f, 1).flatten(1)

    def forward(self, x):
        # Retina + Opsin activation (separate green & blue channel processing)
        p = self.opsin(x)

        # Lamina: apply spatial filtering (center-surround, motion, contrast)
        lam = self.lamina(torch.cat([p, -p], 1))
        lam = self.lamina_norm(lam)
        lam = F.leaky_relu(lam, 0.1)  # Non-linearity

        # Medulla: combine chromatic and achromatic processing
        med_raw = torch.cat([lam,
                            self.med_c(lam),
                            self.med_a(lam.mean(1, keepdim=True).expand_as(lam))], 1)

        # GroupNorm here mimics center-surround antagonism
        med = self.med_n(med_raw)
        med = F.leaky_relu(med, 0.1) 

        # Lobula: integrate complex features
        lob = self.lobula(med)
        lob = self.lobula_norm(lob)
        lob = F.leaky_relu(lob, 0.1)


        # Extract three different VPN pathway features
        vpn = torch.cat([
            self._gp(self.asot(lob[:, :48])),
            self._gp(self.aiot(lob[:, 48:96])),
            self._gp(self.lot(lob[:, 96:]))
        ], dim=1)

        kc_raw = self.kc_p(vpn)
        
        # Apply adaptive sparsity
        kc_sparse = self.sparsity(kc_raw)

        # Mushroom Body: sparsely project into high-dimensional Kenyon Cell space
        return kc_sparse