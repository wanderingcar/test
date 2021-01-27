import math
from Util.icp import *


def lidar_polar_to_xy(dat):
    """
    Switches polar coordinates lidar data to LOCAL cartesian coordinates data
    :param dat: one raw of pandas dataframe of lidar data
    :return: mxN np array
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


def cal_delta_icp(x, y, theta, lidar_0, lidar_1, gps_0, gps_1, previous_T):
    """
    Calculates delta with ICP for each time step
    :param x: current global x
    :param y: current global y
    :param theta: current global theta
    :param lidar_0: lidar data at t with polar coord.
    :param lidar_1: lidar data at t+1 with polar coord.
    :param gps_0: gps data at t
    :param gps_1: gps data at t+1
    :param previous_T: previous T
    :return: updated x, y, theta
    """

    [dx, dy, dt] = np.array(gps_1, dtype=np.float) - np.array(gps_0, dtype=np.float)

    c = math.cos(dt)
    s = math.sin(dt)

    gps_T = np.array([[c, -s, c * dx - s * dy],
                      [s, c, s * dx + c * dy],
                      [0, 0, 1]])

    T, _, _ = icp(lidar_polar_to_xy(lidar_1), lidar_polar_to_xy(lidar_0), init_pose=previous_T)

    # T_c = T[0][0]
    T_s = T[1][0]

    # A = T[0][2]
    # B = T[1][2]
    #
    # x += (T_c * A + T_s * B)
    # y += (-T_s * A + T_c * B)

    pose = np.array([x, y, 1]).reshape(3, 1)

    new_pose = T @ pose

    theta += math.asin(T_s)

    return T, new_pose[0][0], new_pose[1][0], theta
