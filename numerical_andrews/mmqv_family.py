"""The MMQV phase, reflection, and constant-sign family used in the paper."""
# Source repository: https://github.com/n8epi/numericalAndrewsCode
import numpy as np


def mmqv_transform(dimension, angles, reflections, constant_sign):
    """Columns express an orthonormal MMQV basis in cosine-first coordinates.

    The even-dimensional unpaired last cosine is augmented by its sine so
    its phase can vary. Thus the ambient basis dimension is d+1 for even d.
    """
    pairs = (dimension-1)//2
    ambient = dimension + (dimension % 2 == 0)
    transform = np.zeros((ambient, dimension))
    transform[0, 0] = constant_sign
    for pair in range(pairs):
        j = 1+2*pair
        c, s = np.cos(angles[pair]), np.sin(angles[pair])
        h = reflections[pair]
        transform[j:j+2, j:j+2] = [[c, -h*s], [s, h*c]]
    if dimension % 2 == 0:
        c, s = np.cos(angles[-1]), np.sin(angles[-1])
        transform[-2:, -1] = [c, s]
    return transform

