import numpy as np

def tanh(x):
    """Hyperbolic tangent transfer function"""
    return np.tanh(x)
def dtanh(x):
    """Derivative of the tanh function"""
    return 1 - np.tanh(x) ** 2
tanh.d = dtanh

def propagate(x, w, b, f):
    """Propagate the net forward through one level, given the
        inputs (x), weights (w), bias term (b), and transfer function (f).

        """
    return f(np.dot(w, x) + b[:, None])

def error(t, y):
    """Computes the squared error between target values (t)
        and computed values (y).

        """
    return 0.5 * (t - y) ** 2
def derror(t, y):
    """Derivative of the error function."""
    return y - t
error.d = derror

def total_error(x, y, weights, bias):
    """Compute the mean error for multiple inputs. This propagates
        each input through the network, computes the error for each input, 
        and then takes the mean.

        """
    # compute the output of the network
    t = full_propagate(x, weights, bias)
    # compute output error
    return error(y, t).mean()

def update_weights(x, y, weights, bias, alpha):
    """Update the weights of a 2-layer neural network, given the
        inputs (x), the expected outputs (y), a list of the weights at
        each layer (weights), a list of the biases at each layer (bias),
        a list of the transfer functions for each layer (funcs), and the
        learning rate (alpha).

        This updates the weights in `weights` in place.

        """
    (w0, w1) = weights
    (b0, b1) = bias

    for i in range(x.shape[1]):
        # this is the input to the network
        x0 = x[:, [i]]
        # this is the output of the first layer/input to the second layer
        x1 = propagate(x0, w0, b0, tanh)
        # this is the output of the second layer
        x2 = propagate(x1, w1, b1, tanh)

        # compute weight delta for the second layer
        d1 = error.d(y[:, [i]], x2) * tanh.d(x2)
        dw1 = -alpha * np.dot(d1, x1.T)
        db1 = -alpha * d1.sum(axis=1)

        # compute weight delta for first layer
        d0 = np.dot(d1.T, w1).T * tanh.d(x1)
        dw0 = -alpha * np.dot(d0, x0.T)
        db0 = -alpha * d0.sum(axis=1)

        # update weights
        w0 += dw0
        w1 += dw1

        # update bias
        b0 += db0
        b1 += db1
            
def train_multilayer_perceptron(X, Y, num_iters=50):
            
    # include first dimension for Y
    Y = Y[None]

    # learning rate and number of hhidden units
    alpha = 0.001        
    num_hidden = 2
    
    # initialize the weights
    w0 = np.random.randn(num_hidden, X.shape[0])
    w1 = np.random.randn(Y.shape[0], num_hidden)
    weights = (w0, w1)

    # initialize the bias terms
    b0 = np.random.randn(num_hidden)
    b1 = np.random.randn(Y.shape[0])
    bias = (b0, b1)
    
    for i in range(num_iters):
        update_weights(X, Y, weights, bias, alpha)

    return weights, bias


def predict_multilayer_perceptron(X, weights, bias):
    """Fully propagate inputs through the entire neural network,
    given the inputs (X), a list of the weights for each level
    (weights), and a list of the bias terms for each level (bias).
    
    """
    for (w, b) in zip(weights, bias):
        X = tanh(np.dot(w, X) + b[:, None])
    return X[0]