import math


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

    def return_xy(self):
        length = len(self.data)
        x = []
        y = []
        for i in range(length):
            x.append(self.data[i][0])
            y.append(self.data[i][1])

        return x, y
