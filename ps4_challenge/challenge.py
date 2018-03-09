import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.csgraph import shortest_path
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets, decomposition, preprocessing

def swiss_roll(n):
    """
    Generates points in R3 lying on a "Swiss roll" manifold

    Parameters
    ----------
    n : int
        Number of samples to generate

    Returns
    -------
        numpy array of shape (n, 3)
    """
    return datasets.samples_generator.make_swiss_roll(n_samples=n)


def metric_mds(distances):
    """
    Performs metric multidimensional scaling on the given
    distances, projecting the data into a 2D space.

    Parameters
    ----------
    distances : numpy array of shape (n, n)
        The distances between points (i.e., with zeros along the diagonal)

    Returns
    -------
    pos : numpy array of shape (n, 2)
        The coordinates of the data points embedded in a 2D space
    """
    # parameters that are used
    params = {
        'random_state': 23497,
        'eps': 1e-6,
        'max_iter': 500,
        'dissimilarity': "precomputed",
        'n_jobs': 1,
        'n_components': 2
    }

    mds = manifold.MDS(metric=True, n_init=1, **params)
    pos = mds.fit(distances).embedding_

    return pos


def dijkstra(edge_matrix):
    """
    Uses Dijkstra's algorithm to calculate the shortest path matrix for the
    nodes in the network defined by edge_matrix
    
    Parameters
    ----------
    edge_matrix : numpy array of shape (n, n)
        The weights on the edges between nodes in a neighborhood graph. 
        edge_matrix[i,j] = 0 indicates that no edge exists between node
        i and j in the graph.

    Returns
    -------
    dist_matrix : numpy array of shape (n, n)
        The shortest distance between each pair of points in the graph.
    """
    return shortest_path(edge_matrix, method='D')


def standardize(data):
    """
    Standardizes data and returns both the whitened data along with the
    sklearn.preprocessing.StandardScaler object for transforming the 
    data back after fitting. Consult the sklearn documentation for further
    details.
    
    Parameters
    ----------
    data : numpy array of shape (n, m)
        The dataset to standardize. Standardization is performed on the columns
        of the array.
    
    Returns
    -------
    whitened : numpy array of shape (n, m)
        The standardized data. Each column now has a standard deviation of 1 and
        a mean of 0.
    
    scaler : sklearn.preprocessing.StandardScaler object
        A data object holding the mean and standard deviation of the (untransformed) 
        columns in data. To translate the output of the IsoMap algorithm back into
        the units of data, use this object's inverse_transform method. More details 
        can be found in the documentation for sklearn.preprocessing.StandardScaler .
    """
    scaler = preprocessing.StandardScaler().fit(data)
    whitened = scaler.transform(data)
    return whitened, scaler


def symmetrize(edge_matrix):
    """
    Ensures that an edge matrix derived via nearest neighbors is 
    symmetric (a prerequisite for MDS).

    Parameters
    ----------
    edge_matrix : numpy array of shape (n, n)
        The weights on the edges between nodes in a neighborhood graph. 
        edge_matrix[i,j] = 0 indicates that no edge exists between node
        i and j in the graph.
    
    Returns
    -------
    symmetric_edge_matrix : numpy array of shape (n, n)
        An edge matrix symmetric along the main diagonal.
    """
    em = np.triu(edge_matrix)
    return em + em.T


def dijkstra_path(a, b, edge_matrix):
    """
    Returns the nodes in edge_matrix visited on the shortest path between 
    indices a and b.
    
    Parameters
    ----------
    a, b : ints
        The indices in edge_matrix corresponding to the start and end of 
        the path through the network
    edge_matrix : numpy array of shape (n, n)
        The weights on the edges between nodes in a neighborhood graph. 
        edge_matrix[i,j] = 0 indicates that no edge exists between node
        i and j in the graph.
    
    Returns
    -------
    path : numpy array of shape (m,)
        An array of the indices in edge_matrix corresponding to the 
        shortest path between a and b
    """
    dm, pred = shortest_path(edge_matrix, method='D', return_predecessors=True)
    path = []
    while b != a:
        path.append(b)
        b = pred[a, b]
    path.append(a)
    return np.asarray(path)


def pca(data, k=2):
    """
    Return points in data projected along the first k principal components of data. 
    See documentation for sklearn.decomposition.PCA for further details.
    
    Parameters
    ----------
    data : numpy array of shape (n, m)
        The dataset on which to perform PCA
    
    k : int (optional)
        The number of principal components to return. If left unspecified, defaults
        to 2.
    
    Returns
    -------
    embedding : numpy array of shape (n, k)
        The coordinates for the projection of the n points in data along the first k
        principal components of data.
    """
    pca = decomposition.PCA(k)
    pca.fit(data)
    return pca.transform(data)