import numpy as np
from sklearn import manifold
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram as _dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt


def mds(distances):
    """Computes nonmetric multidimensional scaling on the given
    distances, projecting the data into a 2D space.

    Parameters
    ----------
    distances : numpy array of shape (n, n)
        The distances between points (i.e., with zeros along the diagonal)

    Returns
    -------
    numpy array of shape (n, 2)

    """
    # parameters that are used by both of the MDS runs
    params = {
        'random_state': 23497,
        'eps': 1e-6,
        'max_iter': 500,
        'dissimilarity': "precomputed",
        'n_jobs': 1,
        'n_components': 2
    }

    # first fit the metric MDS, which we will use as initialization
    # for the nonmetric MDS algorithm
    mds = manifold.MDS(metric=True, n_init=1, **params)
    pos = mds.fit(distances).embedding_

    # now run the nonmetric MDS algorithm
    nmds = manifold.MDS(metric=False, n_init=1, **params)
    pos = nmds.fit_transform(distances, init=pos)

    return pos


def plot_dendrogram(axis, similarities, labels, colors = 'None'):
    """Computes a hierarchical clustering on the given distances, and
    plots the the hierarchy as a dendrogram. The height of the top of
    each cluster is the distance between its child clusters.

    Parameters
    ----------
    axis : matplotlib axis object
        The axis on which to create the dendrogram  plot
    similarities : numpy array of shape (n, n)
        The similarity between points scaled to be between 0 and 1, with
        1s along the diagonal.
    labels : list with length n
        The labels corresponding to each leaf of the dendrogram

    Returns
    -------
    numpy array of dissimilarities

    """
    lower_diag = squareform(1 - similarities)
    linkage_matrix = linkage(lower_diag)
    _dendrogram(linkage_matrix,
                labels=labels,
                color_threshold=0,
                ax=axis)

    # rotate the axis labels
    for idx, label in enumerate(axis.xaxis.get_ticklabels()):
        label.set_rotation(-90)
        if not colors == 'None':
            # Match this axis label to its corresponding color
            idx = [str(l) for l in labels].index(label.get_text())
            label.set_color(tuple(colors[idx,0:3]))


    return 1 - similarities
