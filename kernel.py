import numpy as np

SOBELX = np.array((
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]), dtype="int")

SOBELY = np.array((
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]), dtype="int")

LAPLACIAN = np.array((
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]), dtype="int")