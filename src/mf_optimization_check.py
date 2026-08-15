import csv
import random
import numpy as np
import torch

from data import make_pairs
from meanflow import MeanFlowNet, meanflow_loss


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# =========================================================
# SETTINGS
# =========================================================

SHRINK = 0.01

SEEDS = [1, 2, 3]

CONFIGS = [
    {"name": "longest", "steps": 12000, "lr": 1e-3},
]

HIDDEN = 128
BATCH_SIZE = 512


# =========================================================
# SEED
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# METRICS
# =========================================================

def reconstruction_mse(pred, true):
    return ((pred - true) ** 2).mean().item()


def spread_ratio(pred, true, labels, centres):

    centre = centres[labels]

    true_dist_sq = (
        (true - centre) ** 2
    ).sum(dim=1)

    pred_dist_sq = (
        (pred - centre) ** 2
    ).sum(dim=1)

    true_spread = torch.sqrt(
        true_dist_sq.mean()
    )

    pred_spread = torch.sqrt(
        pred_dist_sq.mean()
    )

    return (
        pred_spread / true_spread
    ).item()


def alignment_score(pred, true, labels, centres):

    centre = centres[labels]

    true_direction = true - centre
    pred_direction = pred - centre

    dot = (
        true_direction * pred_direction
    ).sum(dim=1)

    true_norm = torch.norm(
        true_direction,
        dim=1
    )

    pred_norm = torch.norm(
        pred_direction,
        dim=1
    )

    cosine = dot / (
        true_norm * pred_norm + 1e-8
    )

    return cosine.mean().item()


# =========================================================
# MEANFLOW BACKWARD SAMPLING
# =========================================================

def sample_meanflow(model, x1, steps):

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

            x = x - (t - r) * u

    return x


# =========================================================
# DATA
# =========================================================

x0_train, x1_train, _, centres = make_pairs(
    n_per_blob=5000,
    shrink=SHRINK,
    seed=0
)

x0_test, x1_test, labels_test, _ = make_pairs(
    n_per_blob=300,
    shrink=SHRINK,
    seed=123
)


x0_train = x0_train.to(device)
x1_train = x1_train.to(device)

x0_test = x0_test.to(device)
x1_test = x1_test.to(device)

labels_test = labels_test.to(device)
centres = centres.to(device)


# =========================================================
# TRAIN ONE MODEL
# =========================================================

def train_one(seed, training_steps, lr):

    set_seed(seed)

    model = MeanFlowNet(
        hidden=HIDDEN
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    n = len(x0_train)

    losses = []

    model.train()

    for step in range(training_steps):

        idx = torch.randint(
            0,
            n,
            (BATCH_SIZE,),
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

        losses.append(loss.item())

    # Loss is noisy, so don't use only the final value
    tail = losses[-200:]

    tail_mean = float(np.mean(tail))
    tail_std = float(np.std(tail))

    return model, tail_mean, tail_std


# =========================================================
# RUN EXPERIMENT
# =========================================================

results = []


for config in CONFIGS:

    print()
    print("=" * 70)

    print(
        f"CONFIG: {config['name']} | "
        f"steps={config['steps']} | "
        f"lr={config['lr']}"
    )

    print("=" * 70)

    for seed in SEEDS:

        print(f"\nSeed {seed}")

        model, loss_mean, loss_std = train_one(
            seed=seed,
            training_steps=config["steps"],
            lr=config["lr"]
        )

        # -----------------------------------------
        # 1-step
        # -----------------------------------------

        pred_1 = sample_meanflow(
            model,
            x1_test,
            steps=1
        )

        mse_1 = reconstruction_mse(
            pred_1,
            x0_test
        )

        spread_1 = spread_ratio(
            pred_1,
            x0_test,
            labels_test,
            centres
        )

        align_1 = alignment_score(
            pred_1,
            x0_test,
            labels_test,
            centres
        )

        # -----------------------------------------
        # 4-step
        # -----------------------------------------

        pred_4 = sample_meanflow(
            model,
            x1_test,
            steps=4
        )

        mse_4 = reconstruction_mse(
            pred_4,
            x0_test
        )

        spread_4 = spread_ratio(
            pred_4,
            x0_test,
            labels_test,
            centres
        )

        align_4 = alignment_score(
            pred_4,
            x0_test,
            labels_test,
            centres
        )

        print(
            f"MF-1 | "
            f"MSE={mse_1:.4f} | "
            f"Spread={spread_1:.4f} | "
            f"Align={align_1:.4f}"
        )

        print(
            f"MF-4 | "
            f"MSE={mse_4:.4f} | "
            f"Spread={spread_4:.4f} | "
            f"Align={align_4:.4f}"
        )

        print(
            f"Tail loss = "
            f"{loss_mean:.4f} ± {loss_std:.4f}"
        )

        results.append({
            "config": config["name"],
            "steps": config["steps"],
            "lr": config["lr"],
            "seed": seed,

            "mf1_mse": mse_1,
            "mf1_spread": spread_1,
            "mf1_alignment": align_1,

            "mf4_mse": mse_4,
            "mf4_spread": spread_4,
            "mf4_alignment": align_4,

            "loss_tail_mean": loss_mean,
            "loss_tail_std": loss_std
        })

        del model
        del pred_1
        del pred_4

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =========================================================
# SAVE RAW RESULTS
# =========================================================

with open(
    "mf_optimization_results.csv",
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


print("\nSaved mf_optimization_results.csv")


# =========================================================
# SUMMARY
# =========================================================

summary = []


for config in CONFIGS:

    rows = [
        r for r in results
        if r["config"] == config["name"]
    ]

    row = {
        "config": config["name"],
        "steps": config["steps"],
        "lr": config["lr"]
    }

    metrics = [
        "mf1_mse",
        "mf1_spread",
        "mf1_alignment",
        "mf4_mse",
        "mf4_spread",
        "mf4_alignment",
        "loss_tail_mean"
    ]

    for metric in metrics:

        values = np.array([
            r[metric] for r in rows
        ])

        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std()

    summary.append(row)


with open(
    "mf_optimization_summary.csv",
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary[0].keys()
    )

    writer.writeheader()
    writer.writerows(summary)


print("Saved mf_optimization_summary.csv")


print()
print("=" * 110)
print("SUMMARY")
print("=" * 110)


for row in summary:

    print(
        f"{row['config']:<15} "
        f"steps={row['steps']:<5} "
        f"lr={row['lr']:<8} | "
        f"MF1 MSE={row['mf1_mse_mean']:.4f} ± {row['mf1_mse_std']:.4f} | "
        f"Spread={row['mf1_spread_mean']:.4f} ± {row['mf1_spread_std']:.4f} | "
        f"Align={row['mf1_alignment_mean']:.4f} ± {row['mf1_alignment_std']:.4f} | "
        f"Loss={row['loss_tail_mean_mean']:.4f}"
    )