"""Visualization utilities matching MATLAB figure outputs."""

import numpy as np
import matplotlib.pyplot as plt


def plot_training_activity(
    PN_activity: np.ndarray,
    KC_activity: np.ndarray,
    EN_activity: np.ndarray,
    save_path: str = None,
):
    """Plot training activity across three phases (Figure 98).

    Matches ant_route_following_test.m:
        figure(98)
        subplot(311): PN activity (green)
        subplot(312): KC activity (black)
        subplot(313): EN activity (red)

    Args:
        PN_activity: (3*numTrain,) PN firing counts per image.
        KC_activity: (3*numTrain,) KC firing counts per image.
        EN_activity: (3*numTrain,) EN firing counts per image.
        save_path: Optional path to save figure.
    """
    n = len(PN_activity)
    x = np.arange(1, n + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    axes[0].plot(x, PN_activity, "gx-", markersize=3, linewidth=0.5)
    axes[0].set_ylabel("PN Activity")
    axes[0].set_title("Network Activity Across Training Phases")

    axes[1].plot(x, KC_activity, "kx-", markersize=3, linewidth=0.5)
    axes[1].set_ylabel("KC Activity")

    axes[2].plot(x, EN_activity, "rx-", markersize=3, linewidth=0.5)
    axes[2].set_ylabel("EN Activity")
    axes[2].set_xlabel("Image Index")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_navigation(
    nav_result: dict,
    world_data: dict = None,
    save_path: str = None,
):
    """Plot navigation result (Figure 97).

    Matches ant_route_following_test.m:
        patch(X', Y', Z')   — terrain
        plot(trained_route) — red training route
        scatter(error_loc)  — green asterisks for errors
        scatter(feeder/nest) — red asterisks
        plot(current_pos)   — blue circles for agent path

    Args:
        nav_result: Dictionary from NavigationAgent.navigate().
        world_data: Optional dict with 'X', 'Y' for terrain rendering.
        save_path: Optional path to save figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 9.5))

    # Draw terrain if world data provided
    if world_data is not None:
        X = world_data["X"]
        Y = world_data["Y"]
        for i in range(X.shape[0]):
            ax.fill(X[i], Y[i], color="black", edgecolor="none")

    ax.axis("off")
    ax.set_aspect("equal")

    # Training route (red line)
    route = nav_result["trained_route"]
    ax.plot(route[:, 0], route[:, 1], "r", linewidth=1.0, label="Route")

    # Error locations (green asterisks)
    if len(nav_result["error_location"]) > 0:
        err = nav_result["error_location"]
        ax.scatter(
            err[:, 0], err[:, 1], marker="*", c="green", s=50, zorder=5,
            label="Errors",
        )

    # Feeder and nest (red asterisks)
    feeder = nav_result["feeder"]
    nest = nav_result["nest"]
    ax.scatter(feeder[0], feeder[1], marker="*", c="red", s=100, zorder=6)
    ax.scatter(nest[0], nest[1], marker="*", c="red", s=100, zorder=6)

    # Agent path (blue circles)
    pos = nav_result["current_position"]
    ax.plot(
        pos[:, 0], pos[:, 1], "bo", linewidth=0.8, markersize=6,
        label="Agent",
    )

    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
