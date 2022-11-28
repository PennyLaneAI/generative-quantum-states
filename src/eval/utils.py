import numpy as np


def get_grid_coordinates(xs, ys):
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.flatten(), yy.flatten()], axis=1)
