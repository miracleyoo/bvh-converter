import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_skeleton_lines(x, y, z):
    """
    From DHP19 toolbox
    """
    # rename joints to identify name and axis
    x_head, x_shoulderR, x_shoulderL, x_elbowR = x[0], x[1], x[2], x[3]
    x_elbowL, x_hipR, x_hipL = (
        x[4],
        x[5],
        x[6],
    )
    x_handR, x_handL, x_kneeR = (
        x[7],
        x[8],
        x[9],
    )
    x_kneeL, x_footR, x_footL = x[10], x[11], x[12]

    y_head, y_shoulderR, y_shoulderL, y_elbowR = y[0], y[1], y[2], y[3]
    y_elbowL, y_hipR, y_hipL = (
        y[4],
        y[5],
        y[6],
    )
    y_handR, y_handL, y_kneeR = (
        y[7],
        y[8],
        y[9],
    )
    y_kneeL, y_footR, y_footL = y[10], y[11], y[12]

    z_head, z_shoulderR, z_shoulderL, z_elbowR = z[0], z[1], z[2], z[3]
    z_elbowL, z_hipR, z_hipL = (
        z[4],
        z[5],
        z[6],
    )
    z_handR, z_handL, z_kneeR = (
        z[7],
        z[8],
        z[9],
    )
    z_kneeL, z_footR, z_footL = z[10], z[11], z[12]

    # definition of the lines of the skeleton graph
    skeleton = np.zeros((14, 3, 2))
    skeleton[0, :, :] = [
        [x_head, x_shoulderR],
        [y_head, y_shoulderR],
        [z_head, z_shoulderR],
    ]
    skeleton[1, :, :] = [
        [x_head, x_shoulderL],
        [y_head, y_shoulderL],
        [z_head, z_shoulderL],
    ]
    skeleton[2, :, :] = [
        [x_elbowR, x_shoulderR],
        [y_elbowR, y_shoulderR],
        [z_elbowR, z_shoulderR],
    ]
    skeleton[3, :, :] = [
        [x_elbowL, x_shoulderL],
        [y_elbowL, y_shoulderL],
        [z_elbowL, z_shoulderL],
    ]
    skeleton[4, :, :] = [
        [x_elbowR, x_handR],
        [y_elbowR, y_handR],
        [z_elbowR, z_handR],
    ]
    skeleton[5, :, :] = [
        [x_elbowL, x_handL],
        [y_elbowL, y_handL],
        [z_elbowL, z_handL],
    ]
    skeleton[6, :, :] = [
        [x_hipR, x_shoulderR],
        [y_hipR, y_shoulderR],
        [z_hipR, z_shoulderR],
    ]
    skeleton[7, :, :] = [
        [x_hipL, x_shoulderL],
        [y_hipL, y_shoulderL],
        [z_hipL, z_shoulderL],
    ]
    skeleton[8, :, :] = [[x_hipR, x_kneeR], [y_hipR, y_kneeR], [z_hipR, z_kneeR]]
    skeleton[9, :, :] = [[x_hipL, x_kneeL], [y_hipL, y_kneeL], [z_hipL, z_kneeL]]
    skeleton[10, :, :] = [
        [x_footR, x_kneeR],
        [y_footR, y_kneeR],
        [z_footR, z_kneeR],
    ]
    skeleton[11, :, :] = [
        [x_footL, x_kneeL],
        [y_footL, y_kneeL],
        [z_footL, z_kneeL],
    ]
    skeleton[12, :, :] = [
        [x_shoulderR, x_shoulderL],
        [y_shoulderR, y_shoulderL],
        [z_shoulderR, z_shoulderL],
    ]
    skeleton[13, :, :] = [[x_hipR, x_hipL], [y_hipR, y_hipL], [z_hipR, z_hipL]]
    return skeleton

def get_3d_ax(ret_fig=False):
    fig = plt.figure(figsize=(8, 8))
    # ax = Axes3D(fig)
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Bonus: To get rid of the grid as well:
    ax.grid(False)
    ax.view_init(30, 240)
    if not ret_fig:
        return ax
    else:
        return ax, fig

def plot_3d(points, ax, c="red", limits=None, plot_lines=True, angle=270, cam_height=10):
    """
    Plot the provided skeletons in 3D coordinate space
    Args:
        ax: axis for plot
        y_true_pred: joints to plot in 3D coordinate space
        c: color (Default value = 'red')
        limits: list of 3 ranges (x, y, and z limits)
        plot_lines:  (Default value = True)

    Note:
        Plot the provided skeletons. Visualization purpose only

    From DHP19 toolbox
    """

    if limits is None:
        limits = [[-3, 3], [-3, 3], [0, 15]]

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    ax.scatter(x, y, z, zdir="z", s=20, c=c, marker="o", depthshade=True)

    lines_skeleton = get_skeleton_lines(x, y, z)

    if plot_lines:
        for line in range(len(lines_skeleton)):
            ax.plot(
                lines_skeleton[line, 0, :],
                lines_skeleton[line, 1, :],
                lines_skeleton[line, 2, :],
                c,
                label="gt",
            )

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    x_limits = limits[0]
    y_limits = limits[1]
    z_limits = limits[2]
    x_range = np.abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = np.abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = np.abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * np.max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    ax.view_init(elev=cam_height, azim=angle)

def plot_skeleton_3d(points, angle=270, cam_height=10, ret_fig=False, limits=None):
    """
        Args:
           M: extrinsic matrix as tensor of shape 4x3
           xyz: torch tensor of shape NUM_JOINTSx3
           pred: torch tensor of shape NUM_JOINTSx3
        """
    ax, fig = get_3d_ax(ret_fig=True)
    plot_3d(points, ax, c='red', angle=angle, cam_height=cam_height, limits=limits)
    if ret_fig:
        return fig