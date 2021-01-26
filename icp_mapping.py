# reference : https://github.com/gisbi-kim/PyICP-SLAM/blob/master/utils/ICP.py

import os
import numpy as np
import pandas as pd
import math
from sklearn.neighbors import NearestNeighbors
from icp import *
import matplotlib.pyplot as plt


class GridMap:
    """
    Global Occupancy Grid Map, stores occupied grid data
    :param grid_size: 0.001, 0.01, 0.1, 1, etc..
    point: tuple type
    """

    def __init__(self, grid_size=0.1):
        self.data = []
        self.size = -int(math.log10(grid_size))

    def is_occupied(self, point):
        point = (round(point[0], self.size), round(point[1], self.size))
        if point in self.data:
            return True
        else:
            return False

    def add_point(self, point):
        point = (round(point[0], self.size), round(point[1], self.size))
        if not self.is_occupied(point):
            self.data.append(point)


def load_lidar():
    """
        Loads lidar data 140106
        :return lidar data pandas dataframe (time filtered)
        """
    path = './Laser_Data/140106/laser_data'
    file_list = os.listdir(path)

    a = list(range(541))
    for i in range(len(a)):
        a[i] = str(a[i])

    rng = ['sec'] + a

    lidar = pd.DataFrame(columns=rng)

    count = 0

    # len(file_list)
    for i in range(len(file_list)):
        new = pd.read_csv(path + '/' + file_list[i], header=None)
        new.columns = rng
        for j in range(len(new)):
            if count == 5 or count == 0:
                lidar = lidar.append(new.iloc[j], ignore_index=True)
                count = 0
            count += 1
        lidar.columns = rng
        print(file_list[i], "loaded")

    return lidar


def load_gps():
    """
    Loads gps data 140106
    :return gps_x: pandas dataframe of gps x coordinate
            gps_y: pandas dataframe of gps y coordinate
            gps_theta: pandas dataframe of gps theta
    """
    gps = pd.read_csv("image_auxilliary.csv")

    gps_x = gps.x
    gps_y = gps.y
    gps_theta = gps.theta

    return gps_x, gps_y, gps_theta


def polar_to_xy(dat):
    """
    Switches polar coordinates to LOCAL euclidean coordinates.
    :param dat: one raw of pandas dataframe of lidar data
    :return: Nxm np array
    """
    x = []
    y = []

    time = dat[0]

    for j in range(1, 542):
        radius = dat[j]
        angle = 5 / 4 * math.pi - (j - 1) / 360 * math.pi
        x.append(radius * math.cos(angle))
        y.append(radius * math.sin(angle))

    result = np.array([x, y])

    return result.T


def cal_delta_icp(x, y, theta, lidar_0, lidar_1, gps_0, gps_1):
    """
    Calculates delta with ICP for each time step
    :param x: current x
    :param y: current y
    :param theta current theta
    :param lidar_0: lidar data at t with polar coord.
    :param lidar_1: lidar data at t+1 with polar coord.
    :param gps_0: gps data at t
    :param gps_1: gps data at t+1
    :return: updated x, y, theta
    """

    [dx, dy, dt] = np.array(gps_1, dtype=np.float) - np.array(gps_0, dtype=np.float)

    c = math.cos(dt)
    s = math.sin(dt)

    gps_T = np.array([[c, s, c * dx + s * dy],
                      [-s, c, -s * dx + c * dy],
                      [0, 0, 1]])

    T, _, _ = icp(polar_to_xy(lidar_0), polar_to_xy(lidar_1), init_pose=gps_T)

    T_c = T[0][0]
    T_s = T[0][1]

    A = T[0][2]
    B = T[1][2]

    x += T_c * A + T_c * B
    y += T_s * A - T_c * B
    theta += math.asin(T_s)

    return x, y, theta


def mapping(x, y, theta, lidar, map):
    """
    adds point at map with rotation transformation
    :param x: pose x
    :param y: pose y
    :param theta: pose theta
    :param lidar: lidar data at t
    :param map: map class
    :return: None
    """

    # load lidar data as local coord.
    x = []
    y = []

    for j in range(1, 542):
        radius = lidar[j]
        if radius != 0:
            angle = 5 / 4 * math.pi - (j - 1) / 360 * math.pi
            x.append(radius * math.cos(angle))
            y.append(radius * math.sin(angle))

    result = np.array([x, y])

    # local to global coord.

    c = math.cos(theta)
    s = math.sin(theta)

    rot_mat = np.array([[c, -s], [s, c]])

    glob = rot_mat @ result + np.array([x, y])

    glob = glob.T

    for i in range(len(glob)):
        p = (glob[i][0], glob[i][1])
        map.add_point(p)


def main():
    map = GridMap(grid_size=0.1)
    # load data
    lidar = load_lidar()
    gps_x, gps_y, gps_theta = load_gps()

    # initial state pose
    global_x = 0
    global_y = 0
    global_theta = 0

    # initial state map
    mapping(global_x, global_y, global_theta, lidar.iloc[0], map)

    print("initial mapping complete")

    length = len(lidar)

    # fig = plt.figure()
    # ax = fig.gca()

    for t in range(len(lidar) - 1):
        # pose update
        global_x, global_y, global_theta = cal_delta_icp(global_x, global_y, global_theta, lidar.iloc[t],
                                                         lidar.iloc[t + 1],
                                                         [gps_x[t], gps_y[t], gps_theta[t]],
                                                         [gps_x[t + 1], gps_y[t + 1], gps_theta[t + 1]])
        print(t, "pose /", length)

        # im = ax.scatter(global_x, global_y, s=0.5)
        # map update
        mapping(global_x, global_y, global_theta, lidar.iloc[t + 1], map)
        print(t, "map /", length)

        print(t, global_x, global_y, global_theta)

    # plot map
    fig = plt.figure()
    ax = fig.gca()
    map_length = len(map.data)
    for i in range(len(map.data)):
        im = ax.scatter(map.data[i][0], map.data[i][1], s=0.5)
        print("plot", i, "/", map_length)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.gca().set_aspect("equal")
    plt.show()


if __name__ == '__main__':
    main()
