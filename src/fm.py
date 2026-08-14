import torch
import torch.nn as nn


class FMNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, x, t):
        if t.ndim == 1:
            t = t.unsqueeze(1)

        inp = torch.cat([x, t], dim=1)
        return self.net(inp)


def fm_loss(model, x0, x1):

    batch_size = x0.shape[0]

    t = torch.rand(
        batch_size, 1,
        device=x0.device
    )

    # Point somewhere between x0 and x1
    xt = (1 - t) * x0 + t * x1

    # True instantaneous velocity
    target_v = x1 - x0

    # Network prediction
    pred_v = model(xt, t)

    loss = ((pred_v - target_v) ** 2).mean()

    return loss