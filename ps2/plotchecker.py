import matplotlib
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose


def get_data(ax):
    lines = ax.get_lines()
    if len(lines) > 0:
        xydata = np.concatenate(
            [x.get_xydata() for x in lines], axis=0)

    else:
        collections = ax.collections
        if len(collections) > 0:
            xydata = np.concatenate(
                [x.get_offsets() for x in collections], axis=0)

        else:
            raise ValueError("no data found")

    return xydata
