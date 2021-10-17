import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


def draw_action(q: np.array, ax):
    cmap = LinearSegmentedColormap.from_list('Custom', ["#ffffd9", "#081d58"], 2)

    sns.heatmap(np.flipud(np.argmax(q, axis=-1)), cmap=cmap, ax=ax, xticklabels=range(1, 12),
                yticklabels=list(reversed(range(11, 22))), cbar_kws = dict(use_gridspec=False,location="bottom", shrink=.6))

    # Get the colorbar object from the Seaborn heatmap
    colorbar = ax.collections[0].colorbar
    # The list comprehension calculates the positions to place the labels to be evenly distributed across the colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + 0.5 * r / (2) + r * i / (2) for i in range(2)])
    colorbar.set_ticklabels(["HIT", "STICK"])
    # ax.tick_params(right=True, left=False, labelright=True, labelleft=False)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()



def draw_value_func(v, ax):
    ax.plot_surface(*np.meshgrid(range(1, 11), range(12, 22)), v,  # DealerShowing, PlayerSum, Value
                    color="White", edgecolor="Gray")
    for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
        axis.line.set_linewidth(0)

    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 0.2))
    ax.set_xticks([1, 10])
    ax.set_yticks([12, 21])
    ax.set_zticks([-1,1])
    ax.grid(False)


def draw_policy(q: np.array, filename=None):
    v = np.max(q, axis=-1)
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')

    draw_action(q[:, :, 1, :], ax1)  # Usable Ace
    ax1.set(title=r"$\pi_{*}$")
    ax1.text(-0.4, 0.5, "Usable\nace", transform=ax1.transAxes,
                    ma="center", size="x-large")
    ax1.set_aspect('equal')

    draw_action(q[:, :, 0, :], ax3)  # No Usable Ace
    ax3.set(xlabel='Dealer showing', ylabel='Player sum')
    ax3.text(-0.4, 0.5, "No\nUsable\nace", transform=ax3.transAxes,
                    ma="center", size="x-large")
    ax3.set_aspect('equal')

    draw_value_func(v[:, :, 1], ax2)  # Usable Ace
    ax2.set(title=r"$V_{*}$")
    draw_value_func(v[:, :, 0], ax4)  # No Usable Ace
    ax4.set(xlabel="Dealer Showing", ylabel="Player sum")

    if filename:
        plt.savefig(filename)
    plt.show()


# draw_policy(model.Q)