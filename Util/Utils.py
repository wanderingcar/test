import math
import numpy as np


def mapping(px, py, theta, lidar, map):
    """
    adds point at map with rotation transformation
    :param px: pose x
    :param py: pose y
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

    glob = rot_mat @ result + np.array([px, py]).reshape(2, 1)

    glob = glob.T

    for i in range(len(glob)):
        p = (glob[i][0], glob[i][1])
        map.add_point(p)
