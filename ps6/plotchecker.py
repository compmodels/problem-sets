import matplotlib
import numpy as np


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


def get_label_text(ax):
    text = [x for x in ax.get_children()
            if isinstance(x, matplotlib.text.Text)]
    text = [x for x in text if x.get_text() != ax.get_title()]
    text = [x for x in text if x.get_text().strip() != '']
    return [x.get_text().strip() for x in text]


def get_label_pos(ax):
    text = [x for x in ax.get_children()
            if isinstance(x, matplotlib.text.Text)]
    text = [x for x in text if x.get_text() != ax.get_title()]
    text = [x for x in text if x.get_text().strip() != '']
    return np.vstack([x.get_position() for x in text])


def get_imshow_data(ax):
    image, = ax.get_images()
    return image._A
