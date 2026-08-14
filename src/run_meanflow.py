import torch
import matplotlib.pyplot as plt

from data import make_pairs
from meanflow import MeanFlowNet, meanflow_loss

import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


def sample_backward(model, x1, steps=1):

    model.eval()

    x = x1.clone()

    with torch.no_grad():

        for k in range(steps):

            t_value = 1.0 - k / steps
            r_value = 1.0 - (k + 1) / steps

            t = torch.full(
                (x.shape[0], 1),
                t_value,
                device=x.device
            )

            r = torch.full(
                (x.shape[0], 1),
                r_value,
                device=x.device
            )

            u = model(x, r, t)

            # MeanFlow update:
            # z_r = z_t - (t-r)u
            x = x - (t - r) * u

    return x


def reconstruction_mse(pred, true):

    return ((pred - true) ** 2).mean().item()


def spread_ratio(pred, true, labels, centres):

    centre = centres[labels]

    true_dist_squared = (
        (true - centre) ** 2
    ).sum(dim=1)

    pred_dist_squared = (
        (pred - centre) ** 2
    ).sum(dim=1)

    true_spread = torch.sqrt(
        true_dist_squared.mean()
    )

    pred_spread = torch.sqrt(
        pred_dist_squared.mean()
    )

    return (
        pred_spread / true_spread
    ).item()


def alignment_score(pred, true, labels, centres):

    centre = centres[labels]

    true_direction = true - centre
    pred_direction = pred - centre

    dot = (true_direction * pred_direction).sum(dim=1)

    true_norm = torch.norm(true_direction, dim=1)
    pred_norm = torch.norm(pred_direction, dim=1)

    cosine = dot / (
        true_norm * pred_norm + 1e-8
    )

    return cosine.mean().item()


# ==================================================
# 1. DATA
# ==================================================

shrink = 0.01

x0_train, x1_train, _, centres = make_pairs(
    n_per_blob=1000,
    shrink=shrink,
    seed=0
)

x0_test, x1_test, labels_test, _ = make_pairs(
    n_per_blob=300,
    shrink=shrink,
    seed=123
)


x0_train = x0_train.to(device)
x1_train = x1_train.to(device)

x0_test = x0_test.to(device)
x1_test = x1_test.to(device)

labels_test = labels_test.to(device)
centres = centres.to(device)


# ==================================================
# 2. MODEL
# ==================================================

model = MeanFlowNet(
    hidden=128
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# ==================================================
# 3. TRAIN
# ==================================================

training_steps = 3000
batch_size = 512

N = len(x0_train)

model.train()


for step in range(training_steps):

    idx = torch.randint(
        0,
        N,
        (batch_size,),
        device=device
    )

    x0 = x0_train[idx]
    x1 = x1_train[idx]

    loss = meanflow_loss(
        model,
        x0,
        x1
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    if step % 300 == 0:

        print(
            f"step {step:4d} | "
            f"loss = {loss.item():.6f}"
        )


# ==================================================
# 4. BACKWARD SAMPLING
# ==================================================

x0_hat_1 = sample_backward(
    model,
    x1_test,
    steps=1
)


x0_hat_4 = sample_backward(
    model,
    x1_test,
    steps=4
)


# ==================================================
# 5. MSE
# ==================================================

mse_1 = reconstruction_mse(
    x0_hat_1,
    x0_test
)


mse_4 = reconstruction_mse(
    x0_hat_4,
    x0_test
)


# ==================================================
# 6. SPREAD RATIO
# ==================================================

spread_1 = spread_ratio(
    x0_hat_1,
    x0_test,
    labels_test,
    centres
)


spread_4 = spread_ratio(
    x0_hat_4,
    x0_test,
    labels_test,
    centres
)


# ==================================================
# 6b. ALIGNMENT
# ==================================================

align_1 = alignment_score(
    x0_hat_1,
    x0_test,
    labels_test,
    centres
)

align_4 = alignment_score(
    x0_hat_4,
    x0_test,
    labels_test,
    centres
)


print("\nResults")
print("----------------------------")

print("MeanFlow 1-step")
print("MSE:", mse_1)
print("Spread ratio:", spread_1)
print("Alignment:", align_1)

print()

print("MeanFlow 4-step")
print("MSE:", mse_4)
print("Spread ratio:", spread_4)
print("Alignment:", align_4)


# ==================================================
# 7. PLOT
# ==================================================

true = x0_test.cpu()
pred1 = x0_hat_1.cpu()
pred4 = x0_hat_4.cpu()
labels = labels_test.cpu()


fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4)
)


axes[0].scatter(
    true[:, 0],
    true[:, 1],
    c=labels,
    s=5,
    cmap="tab10"
)

axes[0].set_title(
    "True x0"
)


axes[1].scatter(
    pred1[:, 0],
    pred1[:, 1],
    c=labels,
    s=5,
    cmap="tab10"
)

axes[1].set_title(
    "MeanFlow backward - 1 step"
)


axes[2].scatter(
    pred4[:, 0],
    pred4[:, 1],
    c=labels,
    s=5,
    cmap="tab10"
)

axes[2].set_title(
    "MeanFlow backward - 4 steps"
)


for ax in axes:

    ax.set_aspect("equal")

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)


plt.tight_layout()

plt.savefig(
    "meanflow_result.png",
    dpi=150
)

plt.show()