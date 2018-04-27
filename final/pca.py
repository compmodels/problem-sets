import numpy as np
from sklearn.decomposition import PCA

def plot_face(axis, face):
    axis.matshow(face.reshape((19, 19)), cmap='gray')
    axis.set_xticks([])
    axis.set_yticks([])

def reconstruct(face, v, num_components, pca):
    projected = np.concatenate([[1], pca.transform(face.reshape(1, -1)).ravel()])
    reconstructed = np.dot(projected[:num_components], v[:num_components])
    return reconstructed
