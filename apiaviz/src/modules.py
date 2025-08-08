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

class SNNVisionModule(nn.Module):
    def __init__(self, kc_dim=1024, lam_ch=12, vpn_ch=64, use_adaptive_kwta=False, beta=0.9):
        super().__init__()
        
        self.beta = beta # Leak rate for all Leaky neurons

        # --- Opsin (Input Current Generation) ---
        self.opsin = nn.Conv2d(2, 6, 1, groups=2, bias=True)

        # --- Lamina ---
        self.lamina = nn.Conv2d(lam_ch, lam_ch, 3, padding=1, padding_mode="reflect", groups=lam_ch, bias=True)
        self.lamina_norm = nn.GroupNorm(num_groups=1, num_channels=lam_ch)
        self.lamina_lif = snn.Leaky(beta=beta, init_hidden=False, threshold=0.1)

        # --- Medulla Pathways (now with individual LIFs) ---
        self.med_c = nn.Conv2d(lam_ch, 2 * lam_ch, 3, padding=1, padding_mode="reflect", groups=2)
        self.med_a = nn.Conv2d(lam_ch, 2 * lam_ch, 3, padding=1, padding_mode="reflect")
        # Add LIF neurons for each pathway
        self.med_c_lif = snn.Leaky(beta=beta, init_hidden=False, threshold=0.5)
        self.med_a_lif = snn.Leaky(beta=beta, init_hidden=False, threshold=0.5)
        
        # With separate LIFs, the single Medulla normalizer and LIF are no longer needed.
        # self.med_n = nn.GroupNorm(12, 60)
        # self.medulla_lif = snn.Leaky(beta=beta, init_hidden=False, threshold=0.5)

        # --- Lobula ---
        # Input channels must now match the concatenated output of Lamina, Med_C, and Med_A spikes
        # lam_ch + 2*lam_ch + 2*lam_ch = 5 * lam_ch = 60
        self.lobula_conv = nn.Conv2d(5 * lam_ch, 128, 5, padding=2, padding_mode="reflect")
        self.lobula_norm = nn.GroupNorm(num_groups=1, num_channels=128)
        self.lobula_lif = snn.Leaky(beta=beta, init_hidden=False, threshold=0.8)

        # --- VPN Pathways (now with individual LIFs) ---
        self.asot = nn.Conv2d(48, vpn_ch, 1)
        self.aiot = nn.Conv2d(48, vpn_ch, 1)
        self.lot  = nn.Conv2d(32, vpn_ch, 1)
        # Add LIF neurons for each VPN pathway
        self.asot_lif = snn.Leaky(beta=beta, init_hidden=False, threshold=0.6)
        self.aiot_lif = snn.Leaky(beta=beta, init_hidden=False, threshold=0.6)
        self.lot_lif = snn.Leaky(beta=beta, init_hidden=False, threshold=0.6)

        # --- Mushroom Body ---
        self.kc_p = SparseLinear(3 * vpn_ch, kc_dim)
        if use_adaptive_kwta:
            self.kc_sparsity = SNNAdaptiveKWTA(sparsity=0.05)
        else:
            self.kc_sparsity = snn.Leaky(beta=beta, init_hidden=False, threshold=1.0)
            
        self._initialize_weights()

    def _initialize_weights(self): # Unchanged
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0.1)
                if m.bias is not None: nn.init.constant_(m.bias, 0.01)

    def _gp(self, f_spikes): # Unchanged
        return F.adaptive_avg_pool2d(f_spikes, 1).flatten(1)

    def forward(self, x, num_steps):
        batch_size = x.size(1)
        
        # Initialize all membrane potentials
        lam_mem = self.lamina_lif.init_leaky()
        med_c_mem = self.med_c_lif.init_leaky()
        med_a_mem = self.med_a_lif.init_leaky()
        lob_mem = self.lobula_lif.init_leaky()
        asot_mem = self.asot_lif.init_leaky()
        aiot_mem = self.aiot_lif.init_leaky()
        lot_mem = self.lot_lif.init_leaky()

        if isinstance(self.kc_sparsity, SNNAdaptiveKWTA):
            kc_mem = torch.zeros(batch_size, self.kc_p.weight.size(0), device=x.device)
        else:
            kc_mem = self.kc_sparsity.init_leaky()
        
        kc_spk_rec = []
        for step in range(num_steps):
            spk_in_step = x[step]
            opsin_cur = self.opsin(spk_in_step)
            
            # 2. Lamina
            lam_cur_in = torch.cat([opsin_cur, -opsin_cur], 1)
            lam_cur = self.lamina_norm(self.lamina(lam_cur_in))
            spk_lam, lam_mem = self.lamina_lif(lam_cur, lam_mem)

            # 3. Medulla (Now fully spiking)
            med_c_cur = self.med_c(spk_lam)
            spk_med_c, med_c_mem = self.med_c_lif(med_c_cur, med_c_mem)
            
            med_a_cur = self.med_a(spk_lam.mean(1, keepdim=True).expand(-1, 12, -1, -1))
            spk_med_a, med_a_mem = self.med_a_lif(med_a_cur, med_a_mem)

            # 4. Lobula (Input is now a concatenation of three spike trains)
            lob_in_spikes = torch.cat([spk_lam, spk_med_c, spk_med_a], 1)
            lob_cur = self.lobula_norm(self.lobula_conv(lob_in_spikes))
            spk_lob, lob_mem = self.lobula_lif(lob_cur, lob_mem)
            
            # 5. VPN pathways (Now fully spiking)
            asot_cur = self.asot(spk_lob[:, :48])
            spk_asot, asot_mem = self.asot_lif(asot_cur, asot_mem)
            
            aiot_cur = self.aiot(spk_lob[:, 48:96])
            spk_aiot, aiot_mem = self.aiot_lif(aiot_cur, aiot_mem)

            lot_cur = self.lot(spk_lob[:, 96:])
            spk_lot, lot_mem = self.lot_lif(lot_cur, lot_mem)
            
            # Pool the VPN spikes
            vpn_spk_pooled = torch.cat([
                self._gp(spk_asot), self._gp(spk_aiot), self._gp(spk_lot)
            ], dim=1)

            # 6. Kenyon Cells
            kc_cur = self.kc_p(vpn_spk_pooled)
            if isinstance(self.kc_sparsity, SNNAdaptiveKWTA):
                kc_mem = self.beta * kc_mem + kc_cur 
                spk_kc, kc_mem = self.kc_sparsity(kc_mem, time_step=step)
            else:
                spk_kc, kc_mem = self.kc_sparsity(kc_cur, kc_mem)

            kc_spk_rec.append(spk_kc)
            
        return torch.stack(kc_spk_rec, dim=0)