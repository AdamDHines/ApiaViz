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
    def __init__(self, kc_dim=1024, lam_ch=12, vpn_ch=64, use_adaptive_kwta=True, beta=0.9):
        super().__init__()
        
        self.beta = beta # Leak rate for all Leaky neurons

        # ───── Retina to Photoreceptor (Opsin response) ─────
        # This layer generates the initial input current, no change needed.
        self.opsin = nn.Conv2d(2, 6, 1, groups=2, bias=True)

        # ───── Lamina (early local motion + contrast detection) ─────
        self.lamina = nn.Conv2d(lam_ch, lam_ch, 3, padding=1, padding_mode="reflect", groups=lam_ch, bias=True)
        self.lamina_norm = nn.GroupNorm(num_groups=1, num_channels=lam_ch)
        self.lamina_lif = snn.Leaky(beta=beta, init_hidden=False)

        # ───── Medulla: Color & Achromatic Pathways ─────
        self.med_c = nn.Conv2d(lam_ch, 2 * lam_ch, 3, padding=1, padding_mode="reflect", groups=2)
        self.med_a = nn.Conv2d(lam_ch, 2 * lam_ch, 3, padding=1, padding_mode="reflect")
        self.med_n = nn.GroupNorm(12, 60)
        self.medulla_lif = snn.Leaky(beta=beta, init_hidden=False)

        # ───── Lobula (higher-order feature integration) ─────
        # We replace the nn.Sequential and ReLU with a Conv layer and a separate Leaky neuron
        self.lobula_conv = nn.Conv2d(60, 128, 5, padding=2, padding_mode="reflect")
        self.lobula_norm = nn.GroupNorm(num_groups=1, num_channels=128)
        self.lobula_lif = snn.Leaky(beta=beta, init_hidden=False)

        # ───── VPN layers: distinct feature projections ─────
        self.asot = nn.Conv2d(48, vpn_ch, 1)
        self.aiot = nn.Conv2d(48, vpn_ch, 1)
        self.lot  = nn.Conv2d(32, vpn_ch, 1)

        # ───── Mushroom Body (Kenyon Cell projection) ─────
        # SparseLinear remains unchanged, it will process VPN spikes into KC currents
        self.kc_p = SparseLinear(3 * vpn_ch, kc_dim)
        
        # Final KC layer with adaptive sparsity
        if use_adaptive_kwta:
            # Our new spiking-aware KWTA layer
            self.kc_sparsity = SNNAdaptiveKWTA(sparsity=0.05)
        else:
            # Default to a standard Leaky neuron if not using adaptive k-WTA
            self.kc_sparsity = snn.Leaky(beta=beta, init_hidden=False)
            
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights to prevent dead neurons. Same as original."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def _gp(self, f_spikes):
        # Global average pooling over spatial dimensions of the SPIKES
        return F.adaptive_avg_pool2d(f_spikes, 1).flatten(1)

    def forward(self, x, num_steps):
        # The input x has shape [T, B, C, H, W] due to our permutation in the training loop.
        # T = num_steps, B = batch_size
        
        # --- Correctly initialize hidden states and membrane potentials ---
        # The batch size is the SECOND dimension of x.
        batch_size = x.size(1)
        
        # Let snn.Leaky handle its own state initialization.
        # We only need to manually initialize the membrane for our custom KWTA layer.
        # The shape must be [batch_size, num_neurons].
        if isinstance(self.kc_sparsity, SNNAdaptiveKWTA):
            kc_mem = torch.zeros(batch_size, self.kc_p.weight.size(0), device=x.device)

        # Initialize membrane potential for the other LIF layers
        # (SNNTorch handles this if init_hidden=True, but doing it explicitly is clearer)
        lam_mem = self.lamina_lif.init_leaky()
        med_mem = self.medulla_lif.init_leaky()
        lob_mem = self.lobula_lif.init_leaky()
        if isinstance(self.kc_sparsity, snn.Leaky):
            kc_mem = self.kc_sparsity.init_leaky()


        # Create a list to record output spikes at each time step
        kc_spk_rec = []

        # --- Start of corrected temporal simulation loop ---
        for step in range(num_steps):
            # Get the input spikes for the current time step.
            # Shape of spk_in_step: [B, C, H, W]
            spk_in_step = x[step]

            # 1. Opsin: Convert input spikes for this step into a current for the next layer.
            # This preserves the original architecture's intent of a 2-to-6 channel mapping.
            opsin_cur = self.opsin(spk_in_step)

            # 2. Lamina: Process current, normalize, and generate spikes.
            # The cat([p, -p]) from the ANN is to model ON/OFF pathways.
            # For an SNN, this can be modeled by separate pathways or simplified.
            # Here, we will preserve the channel doubling to match the layer's expected input size.
            lam_cur_in = torch.cat([opsin_cur, -opsin_cur], 1)
            lam_cur = self.lamina(lam_cur_in)
            lam_cur = self.lamina_norm(lam_cur)
            spk_lam, lam_mem = self.lamina_lif(lam_cur, lam_mem)

            # 3. Medulla: Process lamina spikes and generate medulla spikes
            med_cur_raw = torch.cat([spk_lam,
                                        self.med_c(spk_lam),
                                        self.med_a(spk_lam.mean(1, keepdim=True).expand(-1, 12, -1, -1))], 1)
            med_cur = self.med_n(med_cur_raw)
            spk_med, med_mem = self.medulla_lif(med_cur, med_mem)

            # 4. Lobula: Process medulla spikes and generate lobula spikes
            lob_cur = self.lobula_conv(spk_med)
            lob_cur = self.lobula_norm(lob_cur)
            spk_lob, lob_mem = self.lobula_lif(lob_cur, lob_mem)
            
            # 5. VPN pathways: Process lobula spikes
            vpn_spk = torch.cat([
                self._gp(self.asot(spk_lob[:, :48])),
                self._gp(self.aiot(spk_lob[:, 48:96])),
                self._gp(self.lot(spk_lob[:, 96:]))
            ], dim=1)

            # 6. Kenyon Cells: Project VPN spikes into current
            kc_cur = self.kc_p(vpn_spk) # Shape is now correctly [B, kc_dim] e.g., [7, 1024]

            # 7. Apply sparsity and generate KC spikes
            if isinstance(self.kc_sparsity, SNNAdaptiveKWTA):
                # This operation will now succeed as kc_mem and kc_cur have the same batch size.
                kc_mem = self.beta * kc_mem + kc_cur 
                spk_kc, kc_mem = self.kc_sparsity(kc_mem, time_step=step)
            else: # Standard Leaky neuron
                spk_kc, kc_mem = self.kc_sparsity(kc_cur, kc_mem)

            kc_spk_rec.append(spk_kc)
            
        # Stack recorded spikes over time
        return torch.stack(kc_spk_rec, dim=0)