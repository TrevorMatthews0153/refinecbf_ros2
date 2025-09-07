import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# P = pd.read_csv('truncated_file.csv')

# print(P["y"].min())
# print(P["y"].max())

cbf = np.load('/home/administrator/refine_ws/src/refinecbf_ros2/config/jackal/exp3/cbf_vfs.npy') #import data set

curr_sdf = np.load('/home/administrator/refine_ws/vf.npy')
print(np.unique(cbf[:, :, 0, 5]))
print(np.unique(curr_sdf))

# 1. Create evenly-spaced grid
num_points_x = 51
num_points_y = 51
x_lin = np.linspace(-4, 4, num_points_x)
y_lin = np.linspace(-4, 4, num_points_y)
X_grid, Y_grid = np.meshgrid(x_lin, y_lin, indexing='ij')



# Consistent color scale across both plots
vmin = np.nanmin([np.nanmin(cbf), np.nanmin(cbf)])
vmax = np.nanmax([np.nanmax(cbf), np.nanmax(cbf)])

# # fig, axs = plt.subplots(1, 2, figsize=(26, 8), constrained_layout=True)
# fig, axs = plt.subplots(1, 2, figsize=(24, 8), constrained_layout=True)


# # --- Left: Initial SDF ---
# # ax = axs[0]
# ax = axs[0]
# cs0 = ax.contourf(X_grid, Y_grid, cbf,
#                   levels=50, cmap="jet", alpha=0.7, vmin=vmin, vmax=vmax)
# ax.contour(X_grid, Y_grid, cbf,
#            levels=[0], colors="red", linewidths=4, linestyles="--")
# ax.set_xlabel("$x$ [m]")
# ax.set_ylabel("$y$ [m]")
# ax.set_title("cbf-sdf")
# ax.set_xlim(-4, 4)
# ax.set_ylim(-4, 4)
# ax.grid(True)
# fig.colorbar(cs0, ax=ax)
# ax.xaxis.set_major_locator(plt.MultipleLocator(2))
# ax.yaxis.set_major_locator(plt.MultipleLocator(2))

# # --- Right: Smoothed SDF ---
# ax = axs[1]
# cs1 = ax.contourf(X_grid, Y_grid, curr_sdf,
#                   levels=50, cmap="jet", alpha=0.7, vmin=vmin, vmax=vmax)
# ax.contour(X_grid, Y_grid, curr_sdf,
#            levels=[0], colors="red", linewidths=4, linestyles="--")
# ax.set_xlabel("$x$ [m]")
# ax.set_ylabel("$y$ [m]")
# ax.set_title("Low_res SDF")
# ax.set_xlim(-4, 4)
# ax.set_ylim(-4, 4)
# ax.grid(True)
# fig.colorbar(cs1, ax=ax, label="value function")
# # ax.set_aspect('equal', adjustable='box')
# #add more tick marks
# ax.xaxis.set_major_locator(plt.MultipleLocator(2))
# ax.yaxis.set_major_locator(plt.MultipleLocator(2))

# plt.show()
