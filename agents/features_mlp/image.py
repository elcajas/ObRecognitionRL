
import torch
import torch.nn as nn
import mineclip.utils as U

class ImgFeat(nn.Module):
    def __init__(self, *, output_dim: int = 512, device: torch.device):
        super().__init__()
        self._output_dim = output_dim
        self._device = device
        
        self._mlp = nn.Sequential(
            nn.Linear(in_features=900, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2),
            nn.Flatten()
        )
        
    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, x, **kwargs):
        x = U.any_to_torch_tensor(x, device=self._device)
        return self._mlp(x), None
