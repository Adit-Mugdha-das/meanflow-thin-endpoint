import csv
import random
import numpy as np
import torch

from data import make_pairs
from fm import FMNet, fm_loss
from meanflow import MeanFlowNet, meanflow_loss


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


SHRINKS = [0.10, 0.05, 0.02, 0.01]
SEEDS = [1, 2, 3, 4, 5]

TRAINING_STEPS = 3000
BATCH_SIZE = 512
HIDDEN = 128
LR = 1e-3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def sample_fm(model, x1, steps):

    model.eval()

    x = x1.clone()

    dt = -1.0 / steps

    with torch.no_grad():

        for i in range(steps):

            t_value = 1.0 - i / steps

            t = torch.full(
                (x.shape[0], 1),
                t_value,
                device=x.device
            )

            v = model(x, t)

            x = x + dt * v

    return x


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


def train_fm(x0_train, x1_train, seed):

    set_seed(seed)

    model = FMNet(
        hidden=HIDDEN
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    n = len(x0_train)

    model.train()

    for step in range(TRAINING_STEPS):

        idx = torch.randint(
            0,
            n,
            (BATCH_SIZE,),
            device=device
        )

        x0 = x0_train[idx]
        x1 = x1_train[idx]

        loss = fm_loss(
            model,
            x0,
            x1
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def train_meanflow(x0_train, x1_train, seed):

    set_seed(seed)

    model = MeanFlowNet(
        hidden=HIDDEN
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    n = len(x0_train)

    model.train()

    for step in range(TRAINING_STEPS):

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

    return model


def evaluate(
    name,
    pred,
    true,
    labels,
    centres,
    shrink,
    seed
):

    return {
        "shrink": shrink,
        "seed": seed,
        "method": name,
        "mse": reconstruction_mse(
            pred,
            true
        ),
        "spread": spread_ratio(
            pred,
            true,
            labels,
            centres
        ),
        "alignment": alignment_score(
            pred,
            true,
            labels,
            centres
        )
    }


results = []


for shrink in SHRINKS:

    print()
    print("=" * 50)
    print(f"SHRINK = {shrink}")
    print("=" * 50)

    # Same dataset for all seeds.
    # Seeds only change neural-network training.
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

    for seed in SEEDS:

        print()
        print(f"Shrink {shrink} | Seed {seed}")

        # ==============================
        # FLOW MATCHING
        # ==============================

        print("Training FM...")

        fm_model = train_fm(
            x0_train,
            x1_train,
            seed
        )

        fm_1 = sample_fm(
            fm_model,
            x1_test,
            steps=1
        )

        fm_50 = sample_fm(
            fm_model,
            x1_test,
            steps=50
        )

        result = evaluate(
            "FM-1",
            fm_1,
            x0_test,
            labels_test,
            centres,
            shrink,
            seed
        )

        results.append(result)

        print(
            "FM-1:",
            f"MSE={result['mse']:.4f}",
            f"Spread={result['spread']:.4f}",
            f"Align={result['alignment']:.4f}"
        )

        result = evaluate(
            "FM-50",
            fm_50,
            x0_test,
            labels_test,
            centres,
            shrink,
            seed
        )

        results.append(result)

        print(
            "FM-50:",
            f"MSE={result['mse']:.4f}",
            f"Spread={result['spread']:.4f}",
            f"Align={result['alignment']:.4f}"
        )

        del fm_model
        del fm_1
        del fm_50

        # ==============================
        # MEANFLOW
        # ==============================

        print("Training MeanFlow...")

        mf_model = train_meanflow(
            x0_train,
            x1_train,
            seed
        )

        mf_1 = sample_meanflow(
            mf_model,
            x1_test,
            steps=1
        )

        mf_4 = sample_meanflow(
            mf_model,
            x1_test,
            steps=4
        )

        result = evaluate(
            "MF-1",
            mf_1,
            x0_test,
            labels_test,
            centres,
            shrink,
            seed
        )

        results.append(result)

        print(
            "MF-1:",
            f"MSE={result['mse']:.4f}",
            f"Spread={result['spread']:.4f}",
            f"Align={result['alignment']:.4f}"
        )

        result = evaluate(
            "MF-4",
            mf_4,
            x0_test,
            labels_test,
            centres,
            shrink,
            seed
        )

        results.append(result)

        print(
            "MF-4:",
            f"MSE={result['mse']:.4f}",
            f"Spread={result['spread']:.4f}",
            f"Align={result['alignment']:.4f}"
        )

        del mf_model
        del mf_1
        del mf_4

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==================================================
# SAVE ALL RAW RESULTS
# ==================================================

with open(
    "sweep_results.csv",
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "shrink",
            "seed",
            "method",
            "mse",
            "spread",
            "alignment"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


print()
print("Saved sweep_results.csv")


# ==================================================
# CALCULATE MEAN + STANDARD DEVIATION
# ==================================================

summary = []

methods = [
    "FM-1",
    "FM-50",
    "MF-1",
    "MF-4"
]


for shrink in SHRINKS:

    for method in methods:

        rows = [
            r for r in results
            if r["shrink"] == shrink
            and r["method"] == method
        ]

        mse_values = np.array(
            [r["mse"] for r in rows]
        )

        spread_values = np.array(
            [r["spread"] for r in rows]
        )

        alignment_values = np.array(
            [r["alignment"] for r in rows]
        )

        row = {
            "shrink": shrink,
            "method": method,

            "mse_mean": mse_values.mean(),
            "mse_std": mse_values.std(),

            "spread_mean": spread_values.mean(),
            "spread_std": spread_values.std(),

            "alignment_mean": alignment_values.mean(),
            "alignment_std": alignment_values.std()
        }

        summary.append(row)


with open(
    "sweep_summary.csv",
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "shrink",
            "method",

            "mse_mean",
            "mse_std",

            "spread_mean",
            "spread_std",

            "alignment_mean",
            "alignment_std"
        ]
    )

    writer.writeheader()
    writer.writerows(summary)


print("Saved sweep_summary.csv")


print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)


for row in summary:

    print(
        f"s={row['shrink']:<5} "
        f"{row['method']:<6} | "
        f"MSE {row['mse_mean']:.4f} ± {row['mse_std']:.4f} | "
        f"Spread {row['spread_mean']:.4f} ± {row['spread_std']:.4f} | "
        f"Align {row['alignment_mean']:.4f} ± {row['alignment_std']:.4f}"
    )