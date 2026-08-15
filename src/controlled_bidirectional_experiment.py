import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.func import jvp

try:
    from data import make_pairs
except ImportError as e:
    raise ImportError(
        "Put controlled_bidirectional_experiment.py in the same folder as data.py"
    ) from e


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# MODELS
# =========================================================

class DirectMLP(nn.Module):
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


class RFNet(nn.Module):
    """Instantaneous velocity model v_theta(z_t, t)."""

    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        if t.ndim == 1:
            t = t.unsqueeze(1)
        return self.net(torch.cat([x, t], dim=1))


class MFNet(nn.Module):
    """Average-velocity model u_theta(z_t, r, t)."""

    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, r, t):
        if r.ndim == 1:
            r = r.unsqueeze(1)
        if t.ndim == 1:
            t = t.unsqueeze(1)
        return self.net(torch.cat([x, r, t], dim=1))


class SharedBiMFNet(nn.Module):
    """One MeanFlow network shared by both directions, with a direction bit d."""

    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, r, t, d):
        if r.ndim == 1:
            r = r.unsqueeze(1)
        if t.ndim == 1:
            t = t.unsqueeze(1)
        if d.ndim == 1:
            d = d.unsqueeze(1)
        return self.net(torch.cat([x, r, t, d], dim=1))


# =========================================================
# LOSSES
# =========================================================

def rf_loss(model, source, target):
    b = source.shape[0]
    t = torch.rand(b, 1, device=source.device)
    zt = (1.0 - t) * source + t * target
    v = target - source
    pred = model(zt, t)
    return ((pred - v) ** 2).mean()


def sample_r_t(batch_size, device, diagonal_prob=0.75):
    a = torch.rand(batch_size, 1, device=device)
    b = torch.rand(batch_size, 1, device=device)

    r = torch.minimum(a, b)
    t = torch.maximum(a, b)

    diagonal = torch.rand(batch_size, 1, device=device) < diagonal_prob
    r = torch.where(diagonal, t, r)
    return r, t


def mf_loss(model, source, target):
    """
    Same MeanFlow formulation as the user's current code, generalized to any dimension.
    The path is source at time 0 -> target at time 1.
    Sampling naturally reconstructs source by starting at target and moving 1 -> 0.
    """
    batch_size = source.shape[0]
    r, t = sample_r_t(batch_size, source.device)

    zt = (1.0 - t) * source + t * target
    v = target - source

    def model_fn(z, r_, t_):
        return model(z, r_, t_)

    u, dudt = jvp(
        model_fn,
        (zt, r, t),
        (v, torch.zeros_like(r), torch.ones_like(t)),
    )

    target_u = v - (t - r) * dudt
    target_u = target_u.detach()

    return ((u - target_u) ** 2).mean()


def shared_mf_loss(model, source, target, direction_value):
    batch_size = source.shape[0]
    r, t = sample_r_t(batch_size, source.device)
    d = torch.full((batch_size, 1), float(direction_value), device=source.device)

    zt = (1.0 - t) * source + t * target
    v = target - source

    def model_fn(z, r_, t_, d_):
        return model(z, r_, t_, d_)

    u, dudt = jvp(
        model_fn,
        (zt, r, t, d),
        (v, torch.zeros_like(r), torch.ones_like(t), torch.zeros_like(d)),
    )

    target_u = (v - (t - r) * dudt).detach()
    return ((u - target_u) ** 2).mean()


# =========================================================
# TRAINING
# =========================================================

def train_direct(x_in, x_target, seed, hidden, steps, lr, batch_size):
    set_seed(seed)
    model = DirectMLP(dim=x_in.shape[1], hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(x_in)

    model.train()
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=DEVICE)
        pred = model(x_in[idx])
        loss = ((pred - x_target[idx]) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    return model


def train_rf(source, target, seed, hidden, steps, lr, batch_size):
    set_seed(seed)
    model = RFNet(dim=source.shape[1], hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(source)

    model.train()
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=DEVICE)
        loss = rf_loss(model, source[idx], target[idx])

        opt.zero_grad()
        loss.backward()
        opt.step()

    return model


def train_mf(source, target, seed, hidden, steps, lr, batch_size):
    set_seed(seed)
    model = MFNet(dim=source.shape[1], hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(source)

    model.train()
    losses = []
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=DEVICE)
        loss = mf_loss(model, source[idx], target[idx])

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    tail = losses[-200:] if len(losses) >= 200 else losses
    return model, float(np.mean(tail)), float(np.std(tail))


def train_shared_mf(x0, x1, seed, hidden, steps, lr, batch_size):
    """
    Train one shared network on both tasks:
      d=0: source=x0, target=x1, so sampling reconstructs x0 from x1
      d=1: source=x1, target=x0, so sampling reconstructs x1 from x0
    """
    set_seed(seed)
    model = SharedBiMFNet(dim=x0.shape[1], hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(x0)
    losses = []

    model.train()
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=DEVICE)
        bx0 = x0[idx]
        bx1 = x1[idx]

        loss_back = shared_mf_loss(model, bx0, bx1, direction_value=0.0)
        loss_fwd = shared_mf_loss(model, bx1, bx0, direction_value=1.0)
        loss = 0.5 * (loss_back + loss_fwd)

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    tail = losses[-200:] if len(losses) >= 200 else losses
    return model, float(np.mean(tail)), float(np.std(tail))


# =========================================================
# SAMPLING
# =========================================================

@torch.no_grad()
def sample_rf_forward(model, x0, steps):
    """Integrate the learned RF ODE from time 0 -> 1."""
    model.eval()
    x = x0.clone()
    dt = 1.0 / steps

    for k in range(steps):
        t_value = k / steps
        t = torch.full((len(x), 1), t_value, device=x.device)
        x = x + dt * model(x, t)

    return x


@torch.no_grad()
def sample_rf_backward(model, x1, steps):
    """Integrate the same RF ODE from time 1 -> 0."""
    model.eval()
    x = x1.clone()
    dt = 1.0 / steps

    for k in range(steps):
        t_value = 1.0 - k / steps
        t = torch.full((len(x), 1), t_value, device=x.device)
        x = x - dt * model(x, t)

    return x


@torch.no_grad()
def sample_mf_to_source(model, target_state, steps):
    """
    Standard MeanFlow sampling for a model trained on source -> target.
    Start from target at t=1 and reconstruct source at r=0.
    """
    model.eval()
    x = target_state.clone()

    for k in range(steps):
        t_value = 1.0 - k / steps
        r_value = 1.0 - (k + 1) / steps

        t = torch.full((len(x), 1), t_value, device=x.device)
        r = torch.full((len(x), 1), r_value, device=x.device)

        u = model(x, r, t)
        x = x - (t - r) * u

    return x


@torch.no_grad()
def sample_shared_mf_to_source(model, target_state, steps, direction_value):
    model.eval()
    x = target_state.clone()

    for k in range(steps):
        t_value = 1.0 - k / steps
        r_value = 1.0 - (k + 1) / steps
        t = torch.full((len(x), 1), t_value, device=x.device)
        r = torch.full((len(x), 1), r_value, device=x.device)
        d = torch.full((len(x), 1), float(direction_value), device=x.device)

        u = model(x, r, t, d)
        x = x - (t - r) * u

    return x


# =========================================================
# METRICS FOR CURRENT 2D COMPRESSION TOY
# =========================================================

def reconstruction_mse(pred, true):
    return ((pred - true) ** 2).mean().item()


def spread_ratio(pred, true, labels, centres):
    centre = centres[labels]

    true_spread = torch.sqrt(((true - centre) ** 2).sum(dim=1).mean())
    pred_spread = torch.sqrt(((pred - centre) ** 2).sum(dim=1).mean())

    if true_spread.item() < 1e-12:
        return float("nan")
    return (pred_spread / true_spread).item()


def alignment_score(pred, true, labels, centres):
    centre = centres[labels]
    true_direction = true - centre
    pred_direction = pred - centre

    dot = (true_direction * pred_direction).sum(dim=1)
    true_norm = torch.norm(true_direction, dim=1)
    pred_norm = torch.norm(pred_direction, dim=1)

    cosine = dot / (true_norm * pred_norm + 1e-8)

    valid = true_norm > 1e-8
    if valid.sum() == 0:
        return float("nan")
    return cosine[valid].mean().item()


def compression_metrics(pred, true, labels, centres):
    return {
        "mse": reconstruction_mse(pred, true),
        "spread": spread_ratio(pred, true, labels, centres),
        "alignment": alignment_score(pred, true, labels, centres),
    }


# =========================================================
# INFORMATION-ASYMMETRY TOY
# =========================================================

def make_information_pairs(n, appearance_scale, seed):
    """
    x0 = [m1, m2, a1, a2]
    x1 = [m1, m2, lambda*a1, lambda*a2]

    lambda > 0: invertible but increasingly ill-conditioned.
    lambda = 0: appearance is genuinely removed (many-to-one).
    """
    rng = np.random.default_rng(seed)

    # Anatomy: four 2D clusters so it has visible structure.
    n_blobs = 4
    centres = np.array(
        [[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]],
        dtype=np.float32,
    )
    labels = rng.integers(0, n_blobs, size=n)
    m = centres[labels] + 0.35 * rng.standard_normal((n, 2))

    # Appearance is independent of anatomy in this first clean toy.
    a = rng.standard_normal((n, 2))

    x0 = np.concatenate([m, a], axis=1).astype(np.float32)
    x1 = np.concatenate([m, appearance_scale * a], axis=1).astype(np.float32)

    return torch.tensor(x0), torch.tensor(x1)


def information_metrics(pred, true):
    pred_m = pred[:, :2]
    true_m = true[:, :2]
    pred_a = pred[:, 2:]
    true_a = true[:, 2:]

    full_mse = ((pred - true) ** 2).mean().item()
    anatomy_mse = ((pred_m - true_m) ** 2).mean().item()
    appearance_mse = ((pred_a - true_a) ** 2).mean().item()

    true_spread = torch.sqrt((true_a ** 2).sum(dim=1).mean())
    pred_spread = torch.sqrt((pred_a ** 2).sum(dim=1).mean())

    if true_spread.item() < 1e-12:
        app_spread = float("nan")
    else:
        app_spread = (pred_spread / true_spread).item()

    dot = (pred_a * true_a).sum(dim=1)
    pred_norm = torch.norm(pred_a, dim=1)
    true_norm = torch.norm(true_a, dim=1)
    valid = true_norm > 1e-8

    if valid.sum() == 0:
        app_alignment = float("nan")
    else:
        cosine = dot / (pred_norm * true_norm + 1e-8)
        app_alignment = cosine[valid].mean().item()

    return {
        "mse": full_mse,
        "anatomy_mse": anatomy_mse,
        "appearance_mse": appearance_mse,
        "appearance_spread": app_spread,
        "appearance_alignment": app_alignment,
    }


# =========================================================
# CSV HELPERS
# =========================================================

def save_csv(rows, path):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def make_summary(rows, group_keys, metric_keys):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)

    out = []
    for key, group in groups.items():
        s = {k: v for k, v in zip(group_keys, key)}
        for metric in metric_keys:
            vals = np.array([float(g[metric]) for g in group], dtype=float)
            finite = vals[np.isfinite(vals)]
            if len(finite) == 0:
                s[f"{metric}_mean"] = float("nan")
                s[f"{metric}_std"] = float("nan")
            else:
                s[f"{metric}_mean"] = float(finite.mean())
                s[f"{metric}_std"] = float(finite.std())
        out.append(s)
    return out


# =========================================================
# EXPERIMENT 0: DIRECT MLP CONTROL
# =========================================================

def run_direct(args):
    print("\n=== DIRECT MLP CONTROL: x1 -> x0 at s=0.01 ===")

    x0_train, x1_train, _, centres = make_pairs(
        n_per_blob=args.train_per_blob,
        shrink=0.01,
        seed=0,
    )
    x0_test, x1_test, labels_test, _ = make_pairs(
        n_per_blob=args.test_per_blob,
        shrink=0.01,
        seed=123,
    )

    x0_train = x0_train.to(DEVICE)
    x1_train = x1_train.to(DEVICE)
    x0_test = x0_test.to(DEVICE)
    x1_test = x1_test.to(DEVICE)
    labels_test = labels_test.to(DEVICE)
    centres = centres.to(DEVICE)

    rows = []
    for seed in args.seeds:
        print(f"seed={seed}")
        model = train_direct(
            x1_train,
            x0_train,
            seed,
            args.hidden,
            args.train_steps,
            args.lr,
            args.batch_size,
        )
        model.eval()
        with torch.no_grad():
            pred = model(x1_test)
        m = compression_metrics(pred, x0_test, labels_test, centres)
        rows.append({"seed": seed, **m})
        print(
            f"  MSE={m['mse']:.6f} | Spread={m['spread']:.4f} | "
            f"Align={m['alignment']:.4f}"
        )

    save_csv(rows, Path(args.outdir) / "direct_mlp_results.csv")
    summary = make_summary(rows, [], ["mse", "spread", "alignment"])
    save_csv(summary, Path(args.outdir) / "direct_mlp_summary.csv")


# =========================================================
# EXPERIMENT 1: CURRENT COMPRESSION TOY, RF vs MF
# =========================================================

def run_compression(args):
    print("\n=== COMPRESSION SWEEP: RF vs MF, forward vs backward ===")
    rows = []

    for shrink in args.compression_levels:
        print(f"\n--- shrink={shrink} ---")

        x0_train, x1_train, _, centres = make_pairs(
            n_per_blob=args.train_per_blob,
            shrink=shrink,
            seed=0,
        )
        x0_test, x1_test, labels_test, _ = make_pairs(
            n_per_blob=args.test_per_blob,
            shrink=shrink,
            seed=123,
        )

        x0_train = x0_train.to(DEVICE)
        x1_train = x1_train.to(DEVICE)
        x0_test = x0_test.to(DEVICE)
        x1_test = x1_test.to(DEVICE)
        labels_test = labels_test.to(DEVICE)
        centres = centres.to(DEVICE)

        for seed in args.seeds:
            print(f"seed={seed}: training RF")
            rf = train_rf(
                x0_train, x1_train, seed, args.hidden, args.train_steps, args.lr, args.batch_size
            )

            mf_shared = None
            mf_backward = None
            mf_forward = None
            shared_loss = float("nan")
            mf_back_loss = float("nan")
            mf_fwd_loss = float("nan")

            if args.mf_mode in ["shared", "both"]:
                print(f"seed={seed}: training one SHARED bidirectional MF")
                mf_shared, shared_loss, _ = train_shared_mf(
                    x0_train, x1_train, seed, args.hidden, args.train_steps, args.lr, args.batch_size
                )

            if args.mf_mode in ["separate", "both"]:
                print(f"seed={seed}: training SEPARATE MF for x1 -> x0")
                mf_backward, mf_back_loss, _ = train_mf(
                    x0_train, x1_train, seed, args.hidden, args.train_steps, args.lr, args.batch_size
                )
                print(f"seed={seed}: training SEPARATE MF for x0 -> x1")
                mf_forward, mf_fwd_loss, _ = train_mf(
                    x1_train, x0_train, seed, args.hidden, args.train_steps, args.lr, args.batch_size
                )

            for nfe in args.rf_nfe:
                pred_fwd = sample_rf_forward(rf, x0_test, nfe)
                m = compression_metrics(pred_fwd, x1_test, labels_test, centres)
                rows.append({
                    "condition": shrink, "seed": seed, "method": "RF-shared",
                    "direction": "x0_to_x1", "nfe": nfe,
                    "train_tail_loss": float("nan"), **m,
                })

                pred_back = sample_rf_backward(rf, x1_test, nfe)
                m = compression_metrics(pred_back, x0_test, labels_test, centres)
                rows.append({
                    "condition": shrink, "seed": seed, "method": "RF-shared",
                    "direction": "x1_to_x0", "nfe": nfe,
                    "train_tail_loss": float("nan"), **m,
                })

            for nfe in args.mf_nfe:
                if mf_shared is not None:
                    pred_back = sample_shared_mf_to_source(mf_shared, x1_test, nfe, 0.0)
                    m = compression_metrics(pred_back, x0_test, labels_test, centres)
                    rows.append({
                        "condition": shrink, "seed": seed, "method": "MF-shared",
                        "direction": "x1_to_x0", "nfe": nfe,
                        "train_tail_loss": shared_loss, **m,
                    })

                    pred_fwd = sample_shared_mf_to_source(mf_shared, x0_test, nfe, 1.0)
                    m = compression_metrics(pred_fwd, x1_test, labels_test, centres)
                    rows.append({
                        "condition": shrink, "seed": seed, "method": "MF-shared",
                        "direction": "x0_to_x1", "nfe": nfe,
                        "train_tail_loss": shared_loss, **m,
                    })

                if mf_backward is not None:
                    pred_back = sample_mf_to_source(mf_backward, x1_test, nfe)
                    m = compression_metrics(pred_back, x0_test, labels_test, centres)
                    rows.append({
                        "condition": shrink, "seed": seed, "method": "MF-separate",
                        "direction": "x1_to_x0", "nfe": nfe,
                        "train_tail_loss": mf_back_loss, **m,
                    })

                    pred_fwd = sample_mf_to_source(mf_forward, x0_test, nfe)
                    m = compression_metrics(pred_fwd, x1_test, labels_test, centres)
                    rows.append({
                        "condition": shrink, "seed": seed, "method": "MF-separate",
                        "direction": "x0_to_x1", "nfe": nfe,
                        "train_tail_loss": mf_fwd_loss, **m,
                    })

            del rf
            if mf_shared is not None:
                del mf_shared
            if mf_backward is not None:
                del mf_backward, mf_forward
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    outdir = Path(args.outdir)
    save_csv(rows, outdir / "compression_raw.csv")

    summary = make_summary(
        rows,
        ["condition", "method", "direction", "nfe"],
        ["mse", "spread", "alignment", "train_tail_loss"],
    )
    save_csv(summary, outdir / "compression_summary.csv")


# =========================================================
# EXPERIMENT 2: TRUE INFORMATION ASYMMETRY TOY
# =========================================================

def run_information(args):
    print("\n=== INFORMATION SWEEP: [m,a] <-> [m,lambda*a] ===")
    rows = []

    n_train = args.train_per_blob * 4
    n_test = args.test_per_blob * 4

    for lam in args.information_levels:
        print(f"\n--- lambda={lam} ---")

        x0_train, x1_train = make_information_pairs(n_train, lam, seed=0)
        x0_test, x1_test = make_information_pairs(n_test, lam, seed=123)

        x0_train = x0_train.to(DEVICE)
        x1_train = x1_train.to(DEVICE)
        x0_test = x0_test.to(DEVICE)
        x1_test = x1_test.to(DEVICE)

        for seed in args.seeds:
            print(f"seed={seed}: training RF")
            rf = train_rf(
                x0_train, x1_train, seed, args.hidden, args.train_steps, args.lr, args.batch_size
            )

            mf_shared = None
            mf_backward = None
            mf_forward = None
            shared_loss = float("nan")
            mf_back_loss = float("nan")
            mf_fwd_loss = float("nan")

            if args.mf_mode in ["shared", "both"]:
                print(f"seed={seed}: training one SHARED bidirectional MF")
                mf_shared, shared_loss, _ = train_shared_mf(
                    x0_train, x1_train, seed, args.hidden, args.train_steps, args.lr, args.batch_size
                )

            if args.mf_mode in ["separate", "both"]:
                print(f"seed={seed}: training SEPARATE MF for x1 -> x0")
                mf_backward, mf_back_loss, _ = train_mf(
                    x0_train, x1_train, seed, args.hidden, args.train_steps, args.lr, args.batch_size
                )
                print(f"seed={seed}: training SEPARATE MF for x0 -> x1")
                mf_forward, mf_fwd_loss, _ = train_mf(
                    x1_train, x0_train, seed, args.hidden, args.train_steps, args.lr, args.batch_size
                )

            for nfe in args.rf_nfe:
                pred_fwd = sample_rf_forward(rf, x0_test, nfe)
                m = information_metrics(pred_fwd, x1_test)
                rows.append({
                    "condition": lam, "seed": seed, "method": "RF-shared",
                    "direction": "x0_to_x1", "nfe": nfe,
                    "train_tail_loss": float("nan"), **m,
                })

                pred_back = sample_rf_backward(rf, x1_test, nfe)
                m = information_metrics(pred_back, x0_test)
                rows.append({
                    "condition": lam, "seed": seed, "method": "RF-shared",
                    "direction": "x1_to_x0", "nfe": nfe,
                    "train_tail_loss": float("nan"), **m,
                })

            for nfe in args.mf_nfe:
                if mf_shared is not None:
                    pred_back = sample_shared_mf_to_source(mf_shared, x1_test, nfe, 0.0)
                    m = information_metrics(pred_back, x0_test)
                    rows.append({
                        "condition": lam, "seed": seed, "method": "MF-shared",
                        "direction": "x1_to_x0", "nfe": nfe,
                        "train_tail_loss": shared_loss, **m,
                    })

                    pred_fwd = sample_shared_mf_to_source(mf_shared, x0_test, nfe, 1.0)
                    m = information_metrics(pred_fwd, x1_test)
                    rows.append({
                        "condition": lam, "seed": seed, "method": "MF-shared",
                        "direction": "x0_to_x1", "nfe": nfe,
                        "train_tail_loss": shared_loss, **m,
                    })

                if mf_backward is not None:
                    pred_back = sample_mf_to_source(mf_backward, x1_test, nfe)
                    m = information_metrics(pred_back, x0_test)
                    rows.append({
                        "condition": lam, "seed": seed, "method": "MF-separate",
                        "direction": "x1_to_x0", "nfe": nfe,
                        "train_tail_loss": mf_back_loss, **m,
                    })

                    pred_fwd = sample_mf_to_source(mf_forward, x0_test, nfe)
                    m = information_metrics(pred_fwd, x1_test)
                    rows.append({
                        "condition": lam, "seed": seed, "method": "MF-separate",
                        "direction": "x0_to_x1", "nfe": nfe,
                        "train_tail_loss": mf_fwd_loss, **m,
                    })

            del rf
            if mf_shared is not None:
                del mf_shared
            if mf_backward is not None:
                del mf_backward, mf_forward
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    outdir = Path(args.outdir)
    save_csv(rows, outdir / "information_raw.csv")

    summary = make_summary(
        rows,
        ["condition", "method", "direction", "nfe"],
        [
            "mse",
            "anatomy_mse",
            "appearance_mse",
            "appearance_spread",
            "appearance_alignment",
            "train_tail_loss",
        ],
    )
    save_csv(summary, outdir / "information_summary.csv")


# =========================================================
# MAIN
# =========================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiment",
        choices=["direct", "compression", "information", "all"],
        default="compression",
    )
    p.add_argument("--train-steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--train-per-blob", type=int, default=1000)
    p.add_argument("--test-per-blob", type=int, default=300)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--rf-nfe", type=int, nargs="+", default=[1, 4, 25])
    p.add_argument("--mf-nfe", type=int, nargs="+", default=[1, 4])
    p.add_argument(
        "--mf-mode",
        choices=["shared", "separate", "both"],
        default="shared",
        help="shared = one direction-conditioned MF model; separate = one MF per direction; both = diagnostic ablation",
    )
    p.add_argument(
        "--compression-levels",
        type=float,
        nargs="+",
        default=[0.10, 0.05, 0.02, 0.01],
    )
    p.add_argument(
        "--information-levels",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.1, 0.01, 0.0],
    )
    p.add_argument("--outdir", type=str, default="controlled_results")
    return p.parse_args()


def main():
    args = parse_args()
    print("Device:", DEVICE)
    print("Experiment:", args.experiment)
    print("Training steps per model:", args.train_steps)
    print("Seeds:", args.seeds)

    if args.experiment in ["direct", "all"]:
        run_direct(args)
    if args.experiment in ["compression", "all"]:
        run_compression(args)
    if args.experiment in ["information", "all"]:
        run_information(args)


if __name__ == "__main__":
    main()