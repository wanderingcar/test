import os
import pandas as pd


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
    for i in range(1):
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
