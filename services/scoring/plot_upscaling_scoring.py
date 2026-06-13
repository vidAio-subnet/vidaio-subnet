import os
from pathlib import Path

import numpy as np

from upscaling_scoring import (
    calculate_final_score,
    calculate_length_score,
    calculate_preliminary_score,
    calculate_quality_score,
)


def calculate_score_grid(pieapp_values, content_lengths):
    scores = np.zeros((len(pieapp_values), len(content_lengths)))

    for i, pieapp_score in enumerate(pieapp_values):
        quality_score = calculate_quality_score(float(pieapp_score))
        for j, content_length in enumerate(content_lengths):
            length_score = calculate_length_score(float(content_length))
            preliminary_score = calculate_preliminary_score(quality_score, length_score)
            scores[i, j] = calculate_final_score(preliminary_score)

    return scores


def plot_upscaling_scores(output_path=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pieapp_values = np.linspace(0.0, 2.0, 200)
    length_ranges = [
        (5.0, 10.0, "Current support: 5-10s"),
        (5.0, 320.0, "Full range: 5-320s"),
    ]

    fig, axes = plt.subplots(
        2,
        len(length_ranges),
        figsize=(6 * len(length_ranges), 11),
        gridspec_kw={"height_ratios": [1, 1.2]},
    )

    for col, (min_length, max_length, title) in enumerate(length_ranges):
        content_lengths = np.linspace(min_length, max_length, 200)
        scores = calculate_score_grid(pieapp_values, content_lengths)

        ax_heat = axes[0, col]
        im = ax_heat.imshow(
            scores,
            aspect="auto",
            origin="lower",
            extent=[content_lengths[0], content_lengths[-1], pieapp_values[0], pieapp_values[-1]],
            cmap="viridis",
        )
        ax_heat.set_xlabel("Content Length (s)")
        ax_heat.set_ylabel("PIEAPP Score (lower is better)")
        ax_heat.set_title(title)
        fig.colorbar(im, ax=ax_heat, label="Final Score")

        ax_3d = fig.add_subplot(2, len(length_ranges), len(length_ranges) + col + 1, projection="3d")
        axes[1, col].set_visible(False)

        length_grid, pieapp_grid = np.meshgrid(content_lengths, pieapp_values)
        ax_3d.plot_surface(length_grid, pieapp_grid, scores, cmap="viridis", edgecolor="none", alpha=0.9)
        ax_3d.set_xlabel("Content Length (s)")
        ax_3d.set_ylabel("PIEAPP")
        ax_3d.set_zlabel("Score")
        ax_3d.set_title(title)
        ax_3d.view_init(elev=30, azim=-135)

    fig.suptitle("Upscaling Final Score vs PIEAPP & Content Length", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path is None:
        output_path = Path(__file__).resolve().with_name("upscaling_scoring_plot.png")
    else:
        output_path = Path(output_path)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    out_path = plot_upscaling_scores(os.environ.get("UPSCALING_SCORING_PLOT_PATH"))
    print(f"Upscaling plot saved to {out_path}")
