# reference : https://github.com/gisbi-kim/PyICP-SLAM/blob/master/utils/ICP.py

import matplotlib.pyplot as plt
from Util.gridmap import GridMap
from Util.loadfile import *
from Util.Coord_Util import *
from Util.Utils import *


def main():
    map = GridMap(grid_size=0.1)
    # load data
    lidar = load_lidar()
    gps_x, gps_y, gps_theta = load_gps()

    # initial state pose
    global_x = 0
    global_y = 0
    global_theta = 0

    traj_x = []
    traj_y = []

    # initial state map
    mapping(global_x, global_y, global_theta, lidar.iloc[0], map)

    print("initial mapping complete")

    length = len(lidar) - 2

    fig = plt.figure()
    ax = fig.gca()

    Transform = np.eye(3)

    for t in range(len(lidar) - 1):
        # t+1 pose update
        Transform, global_x, global_y, global_theta = cal_delta_icp(global_x, global_y, global_theta, lidar.iloc[t],
                                                                    lidar.iloc[t + 1],
                                                                    [gps_x[t], gps_y[t], gps_theta[t]],
                                                                    [gps_x[t + 1], gps_y[t + 1], gps_theta[t + 1]],
                                                                    Transform)
        print(t, "pose /", length)

        traj_x.append(global_x)
        traj_y.append(global_y)

        # map update
        mapping(global_x, global_y, global_theta, lidar.iloc[t + 1], map)
        print(t, "map /", length)

        print(t, global_x, global_y, global_theta)

    # plot map
    # fig = plt.figure()
    # ax = fig.gca()
    map_length = len(map.data)
    map_x, map_y = map.return_xy()

    im = ax.scatter(map_x, map_y, s=0.5)
    im = ax.scatter(traj_x, traj_y, c=range(len(traj_x)), cmap='OrRd', s=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.gca().set_aspect("equal")
    plt.show()


if __name__ == '__main__':
    main()
