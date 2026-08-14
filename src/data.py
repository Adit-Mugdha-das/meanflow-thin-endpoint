import numpy as np
import torch


def blob_centres(n_blobs: int, radius: float = 3.0) -> np.ndarray:
    ang = np.linspace(0.0, 2.0 * np.pi, n_blobs, endpoint=False)

    return np.stack(
        [radius * np.cos(ang), radius * np.sin(ang)],
        axis=1
    )


def make_pairs(
    n_per_blob: int = 512,
    n_blobs: int = 4,
    blob_std: float = 0.5,
    shrink: float = 1.0,
    radius: float = 3.0,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    centres = blob_centres(n_blobs, radius)

    labels = np.repeat(
        np.arange(n_blobs),
        n_per_blob
    )

    c = centres[labels]

    # Original source points
    x0 = c + blob_std * rng.standard_normal(
        (len(labels), 2)
    )

    # Compressed endpoint
    x1 = c + shrink * (x0 - c)

    x0 = torch.tensor(x0, dtype=torch.float32)
    x1 = torch.tensor(x1, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)
    centres = torch.tensor(centres, dtype=torch.float32)

    return x0, x1, labels, centres


def plot_pairs(x0, x1, labels, ax=None, title=""):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    x0 = x0.numpy()
    x1 = x1.numpy()
    labels = labels.numpy()

    rng = np.random.default_rng(0)

    idx = rng.choice(
        len(x0),
        size=min(120, len(x0)),
        replace=False
    )

    # Show some x0 -> x1 pairings
    for i in idx:
        ax.plot(
            [x0[i, 0], x1[i, 0]],
            [x0[i, 1], x1[i, 1]],
            color="0.85",
            linewidth=0.5,
            zorder=0
        )

    # Original x0 points
    ax.scatter(
        x0[:, 0],
        x0[:, 1],
        c=labels,
        s=5,
        cmap="tab10",
        label="x0",
        zorder=1
    )

    # Compressed x1 points
    ax.scatter(
        x1[:, 0],
        x1[:, 1],
        c="black",
        s=5,
        label="x1",
        zorder=2
    )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    shrinks = [
        1.0,
        0.5,
        0.25,
        0.1,
        0.05,
        0.01,
        0.0
    ]

    fig, axes = plt.subplots(
        1,
        len(shrinks),
        figsize=(4 * len(shrinks), 4)
    )

    for ax, s in zip(axes, shrinks):

        x0, x1, labels, centres = make_pairs(
            shrink=s,
            n_per_blob=200
        )

        plot_pairs(
            x0,
            x1,
            labels,
            ax=ax,
            title=f"s = {s}"
        )

    plt.tight_layout()

    plt.savefig(
        "data_preview.png",
        dpi=150,
        bbox_inches="tight"
    )

    print("Saved: data_preview.png")

    plt.show()