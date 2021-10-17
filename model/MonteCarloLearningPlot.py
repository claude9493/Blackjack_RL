import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from model.model_visualization import value_function_plot

#%% Load data
vdf_10k = pd.read_csv(os.path.join(sys.path[0], "../record/MC_Learning_10000_episodes.csv"))
vdf_50k = pd.read_csv(os.path.join(sys.path[0], "../record/MC_Learning_50000_episodes.csv"))


#%% Draw
def value_function_plot(vdf, ax_draw):
    # ax_draw.plot_trisurf(vdf.DealerShow, vdf.PlayerSum, vdf.Value,
    #                      color="White", edgecolor='Gray')

    df = vdf.pivot(index='DealerShow', columns='PlayerSum', values='Value')

    ax_draw.plot_surface(*np.meshgrid(df.index, df.columns), df.T,
                         color="White", edgecolor="Gray")

    for axis in [ax_draw.w_xaxis, ax_draw.w_yaxis, ax_draw.w_zaxis]:
        axis.line.set_linewidth(0)

    ax_draw.set_box_aspect((1, 1, 0.4))
    ax_draw.set_xticks([])
    ax_draw.set_yticks([])
    ax_draw.set_zticks([])
    ax_draw.grid(False)
    ax_draw.set_zlim(-1, 1)


fig, ax = plt.subplots(2, 2, figsize=(9, 9), subplot_kw=dict(projection='3d'))
fig.patch.set_facecolor('white')
value_function_plot(vdf_10k[vdf_10k.UsableAce == 1], ax[0][0])
ax[0][0].set(title="After 10,000 episodes")
# ax[0][0].text(x=-1, y=0.58, z=0.5, s="Usable ace", color='black', size=8)
ax[0][0].text2D(-0.2, 0.5, "Usable\nace", transform=ax[0][0].transAxes,
                ma="center", size="x-large")

value_function_plot(vdf_50k[vdf_50k.UsableAce == 1], ax[0][1])
ax[0][1].set(title="After 50,000 episodes")
ax[0][1].set_zticks([-1, 1])

value_function_plot(vdf_10k[vdf_10k.UsableAce == 0], ax[1][0])
# ax[1][0].text(x=-1, y=0.58, z=0.5, s="No Usable ace", color='black', size=8)
ax[1][0].text2D(-0.2, 0.5, "No\nUsable\nace", transform=ax[1][0].transAxes,
                ma="center", size="x-large")

value_function_plot(vdf_50k[vdf_50k.UsableAce == 0], ax[1][1])
ax[1][1].set(xlabel="Dealer Showing", ylabel="Player sum")
ax[1][1].set_xticks([1, 10])
ax[1][1].set_yticks([12, 21])

plt.savefig("Blackjack Value Function after Monte-Carlo Learning.png")
plt.show()
