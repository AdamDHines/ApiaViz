"""Offline fixed-count diagnostic; not the finite-timing navigation model."""
import torch
from .mechanisms import MechanismEncoder


class MatchedCountEncoder(MechanismEncoder):
    @torch.no_grad()
    def forward(self, images):
        output=[]
        for drive in self.currents(images):
            k=round(.05*drive.shape[1])
            winners=torch.argsort(drive,dim=1,descending=True,stable=True)[:,:k]
            if (drive.gather(1,winners)<=0).any():
                raise ValueError("Insufficient positive drives for the fixed-count diagnostic")
            output.append(torch.zeros_like(drive).scatter(1,winners,1.))
        return torch.cat(output,1)
