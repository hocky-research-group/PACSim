from math import acos, cos, pi, sin, sqrt
from typing import Iterator
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation


def generate_fibonacci_sphere_grid_points(number_points: int, radius: float,
                                          random_rotation: bool) -> Iterator[npt.NDArray[np.floating]]:
    """
    Generate points on a sphere using the Fibonacci lattice.

    :param number_points:
        The number of points to generate.
    :type number_points: int
    :param radius:
        The radius of the sphere.
    :type radius: float
    :param random_rotation:
        If True, the points are rotated randomly.
        If False, the points are not rotated.
    :type random_rotation: bool

    :return:
        A generator of the three-dimensional points on the sphere.
    :rtype: Iterator[npt.NDArray[np.floating]]

    :raises ValueError:
        If the number of points is not greater than zero.
        If the radius is not greater than zero.
    """
    # See https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
    if not number_points > 0:
        raise ValueError("The number of points must be greater than zero.")
    if not radius > 0.0:
        raise ValueError("The radius must be greater than zero.")
    golden_ratio = (1.0 + sqrt(5.0)) / 2.0
    epsilon = 0.36
    random_rotation = Rotation.random() if random_rotation else Rotation.identity()
    for i in range(number_points):
        theta = 2.0 * pi * i / golden_ratio
        phi = acos(1.0 - 2.0 * (i + epsilon) / (number_points - 1.0 + 2.0 * epsilon))
        yield random_rotation.apply([cos(theta) * sin(phi) * radius, sin(theta) * sin(phi) * radius, cos(phi) * radius])
