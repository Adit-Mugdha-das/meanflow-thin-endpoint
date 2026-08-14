import torch
import torch.nn as nn
from torch.func import jvp


class MeanFlowNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()

        # x has 2 values + r + t = 4 inputs
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, x, r, t):

        if r.ndim == 1:
            r = r.unsqueeze(1)

        if t.ndim == 1:
            t = t.unsqueeze(1)

        inp = torch.cat([x, r, t], dim=1)

        return self.net(inp)


def sample_r_t(batch_size, device):

    # Sample two random times
    a = torch.rand(batch_size, 1, device=device)
    b = torch.rand(batch_size, 1, device=device)

    # Make sure r <= t
    r = torch.minimum(a, b)
    t = torch.maximum(a, b)

    # 75% diagonal samples: r = t
    diagonal = torch.rand(
        batch_size, 1, device=device
    ) < 0.75

    r = torch.where(diagonal, t, r)

    return r, t


def meanflow_loss(model, x0, x1):

    batch_size = x0.shape[0]
    device = x0.device

    r, t = sample_r_t(
        batch_size,
        device
    )

    # Same straight path used for FM
    zt = (1 - t) * x0 + t * x1

    # Instantaneous conditional velocity
    v = x1 - x0

    def model_fn(z, r_, t_):
        return model(z, r_, t_)

    # MeanFlow JVP:
    # du/dt = v * du/dz + du/dt_explicit
    u, dudt = jvp(
        model_fn,
        (zt, r, t),
        (
            v,
            torch.zeros_like(r),
            torch.ones_like(t)
        )
    )

    # MeanFlow identity
    target = v - (t - r) * dudt

    # Stop gradient through target
    target = target.detach()

    # Plain MSE for our clean toy experiment
    loss = ((u - target) ** 2).mean()

    return loss