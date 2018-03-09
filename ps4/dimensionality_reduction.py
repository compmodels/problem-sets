import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
    
def zscore(X):
    """Normalize an array by shifting it by its mean and scaling it
    by its standard deviation.
    
    """
    return (X - X.mean(axis=0)) / X.std(axis=0)

def generate_A(N=100):
    """Generate dataset A with `N` points."""
    np.random.seed(123)
    x = np.random.randn(N)
    y = x * 3.0 + 5.0
    A_data = zscore(np.array([x, y]).T)
    A_colors = x.copy()
    return A_data, A_colors

def generate_B(N=400):
    """Generate dataset B with `N` points."""
    np.random.seed(123)
    mu = np.array([1, 3, 4])
    C = np.array([[2, 0.8], [0.8, 5], [0.8, 0.8]])
    x = np.random.randn(N, 2)
    B_data = zscore(np.dot(x, C.T) + mu)
    B_colors = B_data[:, 2]
    return B_data, B_colors

def generate_C(N=1000):
    """Generate dataset C with `N` points."""
    np.random.seed(123)
    phi = (3 * np.pi / 2) * (1 + 2 * np.random.rand(N))
    height = 50 * np.random.rand(N)
    C_data = zscore(np.array([2 * phi * np.cos(phi), height, phi * np.sin(phi)]).T)
    C_colors = phi.copy()
    return C_data, C_colors

def plot_3d(X, colors, title):
    """Plot an array in three dimensions."""
    if X.shape[1] == 2:
        X = np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)
    plt.close('all')
    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    axis.scatter(
        X[:, 0], X[:, 1], X[:, 2], 
        c=colors, 
        s=25, 
        vmin=colors.min(), 
        vmax=colors.max(), 
        cmap='rainbow', 
        edgecolor='white')
    axis.set_title(title)

def PCA(X, dims):
    """Perform PCA (principal components analysis)."""
    U, V = np.linalg.eig(np.cov(X.T))
    idx = np.argsort(U)[::-1]
    V = V[:, idx[:dims]]
    return V

def plot_pca(X, dims):
    """Plot the vectors returned by the PCA algorithm."""
    if X.shape[1] == 2:
        X = np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)
    V = PCA(X, dims)
    colors = [(1, 0, 0), (0, 1, 0)]
    for i, v in enumerate(V.T):
        plt.plot([0, v[0]], [0, v[1]], [0, v[2]], linewidth=3, color=colors[i])
        
def plot_lowdim(X, colors, title):
    """Project a dataset to a lower number of dimensions and plot it."""
    plt.close('all')
    V = PCA(X, 2)
    Xp = np.dot(V.T, X.T).T
    fig, axis = plt.subplots()
    axis.scatter(
        Xp[:, 0], Xp[:, 1],
        c=colors, 
        s=25, 
        vmin=colors.min(), 
        vmax=colors.max(), 
        cmap='rainbow', 
        edgecolor='white')
    axis.set_title(title)