import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from itertools import count


def p_multivariate_gaussian(x, mean, covariance):
    """Computes P(x | mean, covariance) where mean and covariance are
    the parameters to a multivariate gaussian distribution.
    
    Parameters
    ----------
    x : numpy array with shape (d,)
        The point for which to compute the probability
    mean : numpy array with shape (d,)
        The mean of the multivariate Gaussian distribution
    covariance : numpy array with shape (d, d)
        The covariance of the multivariate Gaussian distribution
        
    Returns
    -------
    a float corresponding to the probability of x under the multivariate
    Gaussian distribution
    
    """
    if mean.ndim != 1:
        raise ValueError("mean array must be 1-d")
    d = mean.size
    if x.shape != (d,):
        raise ValueError("shape of the data must be ({},), but it is {}".format(d, x.shape))
    if covariance.shape != (d, d):
        raise ValueError("shape of covariance must be ({}, {}), but it is {}".format(d, d, covariance.shape))

    C1 = -0.5 * d * np.log(2 * np.pi)
    C2 = -0.5 * np.linalg.slogdet(covariance)[1]
    exp = -0.5 * np.dot(np.dot(x - mean, np.linalg.inv(covariance)), (x - mean).T)
    return np.exp(C1 + C2 + exp)


def compute_llh(X, cluster_means, cluster_covariances):
    """Computes the log likelihood of the data given the current clusters.
    Note that this assumes all clusters are equally probable.
    
    Parameters
    ----------
    X : numpy array with shape (n, d)
        The x- and y- coordinates of the data points
    cluster_means : numpy array with shape (k, d)
        The mean parameter for each cluster, such that cluster_means[j]
        is the mean for the cluster j.
    cluster_covariances : numpy array with shape (k, d, d)
        The covariance parameter for each cluster, such that
        cluster_covariances[j] is the covariance for cluster j.
        
    Returns
    -------
    the log likelihood of the data X given the current clusters.
        
    """
    p = np.empty((X.shape[0], cluster_means.shape[0]))
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            p[i, j] = p_multivariate_gaussian(X[i], cluster_means[j], cluster_covariances[j])
    p_data = (p / 3).sum(axis=1)
    llh = np.sum(np.log(p_data))
    return llh


def plot_contour(axis, mean, covariance, contour_line=None):
    """Plots the 95% probability mass contour for a multivariate
    Gaussian distribution with mean `mean` and covariance
    `covariance`.
    
    Parameters
    ----------
    axis : matplotlib axis object
        The axis on which to plot the contour
    mean : numpy array with shape (d,)
        The mean of the multivariate Gaussian distribution
    covariance : numpy array with shape (d, d)
        The covariance of the multivariate Gaussian distribution
    contour_line (optional) : matplotlib line object
        An existing contour line to update
        
        
    Returns
    -------
    the matplotlib line object corresponding to the plotted
    contour line
        
    """
    # create points representing the unit circle
    theta = np.linspace(0, 2*np.pi, 360)
    unitcirc = np.array([np.cos(theta), np.sin(theta)])
    
    # compute the appropriate rotation/scale matrix based on
    # the eigenvalues and eigenvectors of the covariance
    eigvals, eigvecs = np.linalg.eig(covariance)
    M = np.dot(eigvecs, np.diag(np.sqrt(eigvals)))
    
    # generate the contour corresponding to the 95% confidence region
    # by transforming the unit circle with the transformation matrix
    # we just computed, and then translating it by the mean
    k = np.sqrt(scipy.stats.chi2.ppf(.95, 2))
    contour = np.dot(k * M, unitcirc).T + mean
    
    # plot the contour and the points
    if contour_line:
        contour_line.set_xdata(contour[:, 0])
        contour_line.set_ydata(contour[:, 1])
    else:
        contour_line, = axis.plot(contour[:, 0], contour[:, 1], color='k', linestyle='-')
    
    return contour_line


def plot_true_clusters(axis, X):
    """Plot the true cluster assignments of data X."""
    axis.plot(X[:10, 0], X[:10, 1], 'co', label='cats')
    axis.plot(X[10:20,0], X[10:20, 1], 'mo', label='dogs')
    axis.plot(X[20:, 0], X[20:, 1], 'ko', label='mops')
    axis.set_title('True Cluster Assignments')
    axis.set_xlim(1, 5)
    axis.set_ylim(1, 5)
    axis.set_aspect('equal')
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(fontsize=10)


def EM_algorithm(X, E_step, M_step, initial_clusters=None):
    """Run the EM algorithm on data X and visualize the
    points and clusters while the algorithm is running.
    
    Parameters
    ----------
    X : numpy array with shape (n, d)
        The x- and y- coordinates of the data points
    E_step : function
        The function to call to execute the "E" step. Should
        return the cluster assignment probabilities.
    M_step : function
        The function to call to execute the "M" step. Should
        return the updated cluster centers and covariances.
    initial_clusters (optional) : numpy array with shape (3, 2)
        The initial cluster centers. If not specified, these
        are generated randomly.
        
    """
        
    # initialize the cluster means and covariances
    if initial_clusters is None:
        cluster_means = np.random.uniform(1, 5, (3, 2))
    else:
        cluster_means = initial_clusters.copy()
    cluster_covariances = np.array([np.eye(2)] * 3) / 100

    # compute the initial probabilities
    p = E_step(X, cluster_means, cluster_covariances)
    old_llh = None
    llh = -100000000

    # plot the contour and the points and the true clusters
    plt.close('all')
    fig, (axis1, axis2) = plt.subplots(1, 2)
    contours = [plot_contour(axis1, cluster_means[i], cluster_covariances[i]) for i in range(3)]
    points = axis1.scatter(X[:, 0], X[:, 1], c=p)
    plot_true_clusters(axis2, X)

    # we will keep looping, running the E and M steps until the log
    # likelihood has converged
    for i in count():
        # update the cluster parameters
        cluster_means, cluster_covariances = M_step(X, p)
        # compute the probability of cluster assignments
        p = E_step(X, cluster_means, cluster_covariances)

        # update the log likelihood of the data
        old_llh = llh
        llh = compute_llh(X, cluster_means, cluster_covariances)
        if np.abs(llh - old_llh) < 1e-5:
            break

        # update the contour lines and the point colors
        for j in range(3):
            plot_contour(axis1, cluster_means[j], cluster_covariances[j], contours[j])
            points.set_color(p)

        # update the plot title, axis limits, etc.
        axis1.set_title("EM iteration {} (LLH={:.4})".format(i + 1, llh))
        axis1.set_xlim(1, 5)
        axis1.set_ylim(1, 5)
        axis1.set_aspect('equal')
        axis1.set_xticks([])
        axis1.set_yticks([])
        plt.draw()
