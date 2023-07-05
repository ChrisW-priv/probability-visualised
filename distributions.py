import numpy as np
import math
from typing import Callable


def dist_normal(mu: float, sigma: float):
    """N(mu, sigma^2)"""
    def handle(x: float):
        scale = 1/np.sqrt(2*np.pi*sigma)
        shape_shift = (x-mu)**2/(2*sigma)
        exponent = np.exp(-shape_shift/2)
        return scale * exponent
    v_handle = np.vectorize(handle, otypes=[float])
    return v_handle


def dist_poisson(lambda_p: float):
    """Poiss(lambda)"""
    def handle(k: int):
        shape = lambda_p**k / np.math.factorial(k)
        exponent = np.exp(-lambda_p)
        return shape * exponent
    v_handle = np.vectorize(handle, otypes=[float])
    return v_handle


def dist_uniform(a: float, b: float):
    """U(a,b)
    Warning: a has to be smaller than b!
    """
    assert a < b

    def handle(x: int):
        return 1/(b-a) * (a <= x and x <= b)
    v_handle = np.vectorize(handle, otypes=[float])
    return v_handle


def dist_binomial(n: int, p: float):
    """bin(n, p)"""
    def handle(k):
        return math.comb(n, k) * p**k * (1-p)**(n-k)
    v_handle = np.vectorize(handle, otypes=[float])
    return v_handle


def dist_geometric(p: float):
    """geom(p)"""
    def handle(n):
        return (1-p)**(n-1) * p
    v_handle = np.vectorize(handle, otypes=[float])
    return v_handle


def rand_var_transform(sx: np.ndarray, x: np.ndarray,
                       g: Callable[[np.ndarray], np.ndarray]):
    g_sx = g(sx).round(6)
    d_sy = {}
    for i in range(g_sx.size):
        d_sy[g_sx[i]] = d_sy.get(g_sx[i], 0) + x[i]
    sy = np.fromiter((a for a in d_sy), float)
    sy = np.sort(sy)
    y = np.fromiter((d_sy[a] for a in sy), float)
    return sy, y


def discrete_cdf(x: np.ndarray): return np.cumsum(x) / np.sum(x)


def dist_multivariable_normal(m, c):
    def handle(x):
        denominator = (2*np.pi)**(m.size/2) * np.sqrt(np.linalg.det(c))
        inv_c = np.linalg.inv(c)
        result = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            sub_t = (x[i]-m).transpose()
            foo = np.matmul(sub_t, inv_c)
            sub = np.subtract(x[i], m)
            baz = np.matmul(foo, sub)
            exponent = -baz/2
            result[i] = 1/denominator * np.exp(exponent)
        return result
    return handle


def cartesian_prod(x, y):
    return np.transpose(
        [np.tile(x, len(y)),
         np.repeat(y, len(x))])
