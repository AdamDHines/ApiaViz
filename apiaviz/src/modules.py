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
        self.opsin = nn.Conv2d(2, 6, 1, groups=2, bias=True)

        # ───── Lamina (early local motion + contrast detection) ─────
        self.lamina = nn.Conv2d(lam_ch, lam_ch, 3, padding=1, padding_mode="reflect", groups=lam_ch, bias=True)
        # Replaced GroupNorm with LayerNorm
        self.lamina_norm = nn.LayerNorm(lam_ch)

        # ───── Medulla: Color & Achromatic Pathways ─────
        self.med_c = nn.Conv2d(lam_ch, 2 * lam_ch, 3, padding=1, padding_mode="reflect", groups=2)
        self.med_a = nn.Conv2d(lam_ch, 2 * lam_ch, 3, padding=1, padding_mode="reflect")
        # Replaced GroupNorm with LayerNorm. This changes the normalization strategy from grouped to full.
        self.med_n = nn.LayerNorm(60) # 2*lam_ch + 2*lam_ch + lam_ch assuming concatenation from med_c and med_a

        # ───── Lobula (higher-order feature integration) ─────
        self.lobula = nn.Conv2d(60, 128, 5, padding=2, padding_mode="reflect")
        # Replaced GroupNorm with LayerNorm
        self.lobula_norm = nn.LayerNorm(128)

        # ───── VPN layers: distinct feature projections ─────
        self.asot = nn.Conv2d(48, vpn_ch, 1)
        self.aiot = nn.Conv2d(48, vpn_ch, 1)
        self.lot  = nn.Conv2d(32, vpn_ch, 1)

        # ───── Mushroom Body (Kenyon Cell projection) ─────
        self.kc_p = SparseLinear(3 * vpn_ch, kc_dim)

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
        lam = lam.permute(0, 2, 3, 1)
        lam = self.lamina_norm(lam)
        lam = lam.permute(0, 3, 1, 2)
        lam = F.leaky_relu(lam, 0.1)

        # Medulla: combine chromatic and achromatic processing
        med_raw = torch.cat([lam,
                            self.med_c(lam),
                            self.med_a(lam.mean(1, keepdim=True).expand_as(lam))], 1)

        # Apply LayerNorm to Medulla features
        med = med_raw.permute(0, 2, 3, 1)
        med = self.med_n(med)
        med = med.permute(0, 3, 1, 2)
        med = F.leaky_relu(med, 0.1) 

        # Lobula: integrate complex features
        lob = self.lobula(med)
        # Apply LayerNorm to Lobula features
        lob = lob.permute(0, 2, 3, 1)
        lob = self.lobula_norm(lob)
        lob = lob.permute(0, 3, 1, 2)
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
        # This layer outputs a current, not spikes.
        self.opsin = nn.Conv2d(2, 6, 1, groups=2, bias=True)

        # ───── Lamina (early local motion + contrast detection) ─────
        self.lamina = nn.Conv2d(lam_ch, lam_ch, 3, padding=1, padding_mode="reflect", groups=lam_ch, bias=True)
        self.lamina_norm = nn.LayerNorm(lam_ch)
        self.lamina_lif = snn.Leaky(beta=beta, init_hidden=False, threshold=0.1)

        # ───── Medulla: Color & Achromatic Pathways ─────
        self.med_c = nn.Conv2d(lam_ch, 2 * lam_ch, 3, padding=1, padding_mode="reflect", groups=2)
        self.med_a = nn.Conv2d(lam_ch, 2 * lam_ch, 3, padding=1, padding_mode="reflect")
        # A single normalization and LIF layer for the combined medulla input
        self.med_n = nn.LayerNorm(5 * lam_ch) # lam + med_c + med_a = 12 + 24 + 24 = 60
        self.med_lif = snn.Leaky(beta=beta, init_hidden=False, threshold=0.5)

        # ───── Lobula (higher-order feature integration) ─────
        self.lobula_conv = nn.Conv2d(5 * lam_ch, 128, 5, padding=2, padding_mode="reflect")
        self.lobula_norm = nn.LayerNorm(128)
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
        # Global average pooling over spatial dimensions
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
            kc_mem = torch.zeros(x.size(1), self.kc_p.linear.out_features, device=x.device)
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
            # Apply LayerNorm on current: Permute (N,C,H,W)->(N,H,W,C), normalize, permute back
            lam_cur_permuted = lam_cur.permute(0, 2, 3, 1)
            lam_norm_cur = self.lamina_norm(lam_cur_permuted).permute(0, 3, 1, 2)
            spk_lam, lam_mem = self.lamina_lif(lam_norm_cur, lam_mem)

            # 3. Medulla: Process spikes from Lamina, combine, Norm, Spikes
            # Generate currents from the two medulla pathways using lamina spikes
            med_c_cur = self.med_c(spk_lam)
            ach_in = spk_lam.mean(1, keepdim=True).expand_as(spk_lam)
            med_a_cur = self.med_a(ach_in)
            # Concatenate lamina spikes and pathway currents to form the input, matching the ANN
            med_raw_cur = torch.cat([spk_lam, med_c_cur, med_a_cur], 1)
            # Apply LayerNorm on the combined current
            med_raw_cur_permuted = med_raw_cur.permute(0, 2, 3, 1)
            med_norm_cur = self.med_n(med_raw_cur_permuted).permute(0, 3, 1, 2)
            spk_med, med_mem = self.med_lif(med_norm_cur, med_mem)

            # 4. Lobula: Conv on Medulla spikes, Norm, Spikes
            lob_cur = self.lobula_conv(spk_med)
            # Apply LayerNorm on current
            lob_cur_permuted = lob_cur.permute(0, 2, 3, 1)
            lob_norm_cur = self.lobula_norm(lob_cur_permuted).permute(0, 3, 1, 2)
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
                # SNNAdaptiveKWTA handles its own membrane update
                spk_kc, kc_mem = self.kc_sparsity(kc_mem + kc_cur, time_step=step)
            else:
                spk_kc, kc_mem = self.kc_sparsity(kc_cur, kc_mem)

            kc_spk_rec.append(spk_kc)

        return torch.stack(kc_spk_rec, dim=0)