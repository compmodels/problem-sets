import numpy as np
from IPython.display import display_latex


def format_equation(theta):
    """Formats a LaTeX equation for the polynomial with
    coefficients `theta`, where theta[-1] is the lowest
    order coefficient (the constant) and theta[0] is the
    highest order coefficient.
    
    Parameters
    ----------
    theta : numpy array with shape (k+1,)
        The coefficients of the polynomial, ordered from
        highest-order to lowest-order.
    
    """
    terms = []
    for i in range(len(theta)):
        if theta[i] < 0:
            terms.append("-")
        elif i > 0:
            terms.append("+")

        fmt = "{:.2f}".format(np.abs(theta[i]))
        if i < (len(theta) - 1):
            fmt += "x"
        if i < (len(theta) - 2):
            fmt += "^{" + "{:d}".format(len(theta) - 1 - i) + "}"

        terms.append(fmt)
        
    equation = r"g(x)=" + "".join(terms)
    return equation

    
def print_equation(theta):
    """Displays a LaTeX equation for the polynomial with
    coefficients `theta`, where theta[-1] is the lowest
    order coefficient (the constant) and theta[0] is the
    highest order coefficient.
    
    Parameters
    ----------
    theta : numpy array with shape (k+1,)
        The coefficients of the polynomial, ordered from
        highest-order to lowest-order.
    
    """
    display_latex("$${}$$".format(format_equation(theta)), raw=True)
    