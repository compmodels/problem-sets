import numpy as np
from plot_perceptron import calc_normal, init_plot, sigmoid


def train_perceptron(X, Y, bias, num_loops=100, alpha=0.5, plot=False):
    """ Based on code from from
    http://www.cs.cmu.edu/afs/cs/academic/class/15782-f06/matlab/perceptron/.
    """
    num_patterns = X.shape[1]
    inputs = np.vstack((np.ones((1, num_patterns)), X))
    weights = np.array([bias, -9, -9], dtype=float)

    x = X.T
    y = Y.T

    # initialize helper variables
    X = inputs
    errors = np.empty(num_loops * y.size)

    # set up our initial weights and normal vectors
    normal, boundary = calc_normal(np.random.randn(2), weights)

    # initialize the plot
    if plot:
        update_plot = init_plot(x, y, boundary, num_loops)

    for i in range(num_loops):
        # update the weights
        weights += np.dot(Y - np.tanh(np.dot(weights, inputs)), inputs.T)

        # compute the output of the perceptron and the error to the true labels
        output = sigmoid(np.dot(weights, X))
        errors[i] = ((y - output) ** 2).sum()

        # recalculate the normal vector
        normal, boundary = calc_normal(normal, weights)

        # update the plot
        if plot:
            update_plot(errors[:(i+1)], boundary)

        # update the learning rate
        alpha = alpha * 0.8


    return (weights[1:], weights[0])