# Imports
import torch

import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F

from .functional import AdaptiveKWTA, SNNAdaptiveKWTA, k_wta, SparseLinear

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
        # Retina + Opsin activation
        p = self.opsin(x)

        # Lamina: apply spatial filtering
        lam = self.lamina(torch.cat([p, -p], 1))
        # Apply LayerNorm: Permute from (N, C, H, W) to (N, H, W, C) and back
        lam = self.lamina_norm(lam)
        lam = F.leaky_relu(lam, 0.1)

        # Medulla: combine chromatic and achromatic processing
        med_raw = torch.cat([lam,
                            self.med_c(lam),
                            self.med_a(lam.mean(1, keepdim=True).expand_as(lam))], 1)

        # Apply LayerNorm to Medulla features
        med = self.med_n(med_raw)
        med = F.leaky_relu(med, 0.1) 

        # Lobula: integrate complex features
        lob = self.lobula(med)
        # Apply LayerNorm to Lobula features
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

        return kc_sparse

class SNNVisionModule(nn.Module):
    def __init__(self, kc_dim=1024, lam_ch=12, vpn_ch=64, use_adaptive_kwta=False, beta=0.9):
        super().__init__()

        self.beta = beta
        self.lam_ch = lam_ch

        # ───── Retina to Photoreceptor (Opsin response) ─────
        self.opsin = nn.Conv2d(2, 6, 1, groups=2, bias=True)

        # ───── Lamina (early local motion + contrast detection) ─────
        self.lamina = nn.Conv2d(lam_ch, lam_ch, 3, padding=1, padding_mode="reflect", groups=lam_ch, bias=True)
        self.lamina_norm = nn.GroupNorm(num_groups=1, num_channels=lam_ch)
        self.lamina_lif = snn.Leaky(beta=beta, init_hidden=False, threshold=0.1)

        # ───── Medulla: Color & Achromatic Pathways ─────
        self.med_c = nn.Conv2d(lam_ch, 2 * lam_ch, 3, padding=1, padding_mode="reflect", groups=2)
        self.med_a = nn.Conv2d(lam_ch, 2 * lam_ch, 3, padding=1, padding_mode="reflect")
        
        # FIX: Input to medulla LIF is only from med_c and med_a currents.
        medulla_channels = 4 * lam_ch # 2*lam_ch + 2*lam_ch = 24 + 24 = 48
        
        # FIX: Use multiple groups to preserve the original model's architectural intent.
        # This preserves distinct feature pathways, which a single group would merge.
        # 8 groups for 48 channels (6 ch/group) is analogous to 12 groups for 60 channels.
        self.med_n = nn.GroupNorm(num_groups=8, num_channels=medulla_channels)
        self.med_lif = snn.Leaky(beta=beta, init_hidden=False, threshold=0.5)

        # ───── Lobula (higher-order feature integration) ─────
        # FIX: The input channel count must match the output of the medulla layer.
        self.lobula_conv = nn.Sequential(
            nn.Conv2d(60, 128, 5, padding=2, padding_mode="reflect"),
            nn.ReLU()
        )
        self.lobula_norm = nn.GroupNorm(num_groups=1, num_channels=128)
        self.lobula_lif = snn.Leaky(beta=beta, init_hidden=False, threshold=0.5)

        # ───── VPN layers: distinct feature projections ─────
        self.asot = nn.Conv2d(48, vpn_ch, 1)
        self.aiot = nn.Conv2d(48, vpn_ch, 1)
        self.lot  = nn.Conv2d(32, vpn_ch, 1)
        self.asot_lif = snn.Leaky(beta=beta, init_hidden=False, threshold=0.4)
        self.aiot_lif = snn.Leaky(beta=beta, init_hidden=False, threshold=0.4)
        self.lot_lif = snn.Leaky(beta=beta, init_hidden=False, threshold=0.4)

        # ───── Mushroom Body (Kenyon Cell projection) ─────
        self.kc_p = SparseLinear(3 * vpn_ch, kc_dim)
        if use_adaptive_kwta:
            self.kc_sparsity = SNNAdaptiveKWTA(sparsity=0.05, beta=beta)
        else:
            self.kc_sparsity = snn.Leaky(beta=beta, init_hidden=False, threshold=0.7)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights to prevent dead neurons."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def _gp(self, f_spikes):
        """Global average pooling helper."""
        return F.adaptive_avg_pool2d(f_spikes, 1).flatten(1)

    def forward(self, x, num_steps):
        # Initialize membrane potentials for all LIF layers
        lam_mem = self.lamina_lif.init_leaky()
        med_mem = self.med_lif.init_leaky()
        lob_mem = self.lobula_lif.init_leaky()
        asot_mem = self.asot_lif.init_leaky()
        aiot_mem = self.aiot_lif.init_leaky()
        lot_mem = self.lot_lif.init_leaky()

        if isinstance(self.kc_sparsity, SNNAdaptiveKWTA):
             # Custom init for stateful adaptive layer
            kc_mem = torch.zeros(x.shape[1], self.kc_p.linear.out_features, device=x.device)
        else:
            kc_mem = self.kc_sparsity.init_leaky()

        kc_spk_rec = []
        for step in range(num_steps):
            spk_in_step = x[step]

            # 1. Opsin Current Generation
            opsin_cur = self.opsin(spk_in_step)

            # 2. Lamina: ON/OFF channels, Conv, Norm, Spikes
            lam_in = torch.cat([opsin_cur, -opsin_cur], 1)
            lam_cur = self.lamina(lam_in)
            # FIX: Cleaned up unnecessary variables. GroupNorm doesn't need permutation.
            lam_norm_cur = self.lamina_norm(lam_cur)
            spk_lam, lam_mem = self.lamina_lif(lam_norm_cur, lam_mem)

            # 3. Medulla: Process spikes from Lamina, combine currents, Norm, Spikes
            med_c_cur = self.med_c(spk_lam)
            ach_in = spk_lam.mean(1, keepdim=True).expand_as(spk_lam)
            med_a_cur = self.med_a(ach_in)

            med_raw_cur = torch.cat([med_c_cur, med_a_cur], 1)
            
            med_norm_cur = self.med_n(med_raw_cur)
            spk_med, med_mem = self.med_lif(med_norm_cur, med_mem)

            # 4. Lobula: Conv on Medulla spikes, Norm, Spikes
            lob_cur = self.lobula_conv(spk_med)
            lob_norm_cur = self.lobula_norm(lob_cur)
            spk_lob, lob_mem = self.lobula_lif(lob_norm_cur, lob_mem)

            # 5. VPN pathways: Process slices of Lobula spikes
            asot_cur = self.asot(spk_lob[:, :48])
            spk_asot, asot_mem = self.asot_lif(asot_cur, asot_mem)

            aiot_cur = self.aiot(spk_lob[:, 48:96])
            spk_aiot, aiot_mem = self.aiot_lif(aiot_cur, aiot_mem)

            lot_cur = self.lot(spk_lob[:, 96:])
            spk_lot, lot_mem = self.lot_lif(lot_cur, lot_mem)

            vpn_spk_pooled = torch.cat([self._gp(spk_asot), self._gp(spk_aiot), self._gp(spk_lot)], dim=1)

            # 6. Kenyon Cells: Sparse projection and spiking
            kc_cur = self.kc_p(vpn_spk_pooled)
            if isinstance(self.kc_sparsity, SNNAdaptiveKWTA):
                spk_kc, kc_mem = self.kc_sparsity(kc_mem + kc_cur, time_step=step)
            else:
                spk_kc, kc_mem = self.kc_sparsity(kc_cur, kc_mem)

            kc_spk_rec.append(spk_kc)

        return torch.stack(kc_spk_rec, dim=0)