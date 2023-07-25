import functools
import scipy as sp
from scipy.special import loggamma

import numpy as np

"""

X and Y must have same length

"""


def marginal_prob(X):
    """
    """
    marginals = {x: len(X[X == x]) / len(X) for x in X.unique()}
    return marginals


# TODO n-ary functions
def joint_prob_2(X, Y):
    joints = {(x, y): len(X[(X == x) & (Y == y)]) / len(X)
              for x in X.unique()
              for y in Y.unique()}
    return joints


def joint_prob_3(X, Y, Z):
    joints = {(x, y, z): len(X[(X == x) & (Y == y) & (Z == z)]) / len(X)
              for x in X.unique()
              for y in Y.unique()
              for z in Z.unique()}
    return joints


def joint_prob(*Xs):
    if len(Xs) == 2:
        return joint_prob_2(*Xs)
    elif len(Xs) == 3:
        return joint_prob_3(*Xs)
    else:
        raise NotImplementedError


def mutual_information(X, Y):
    marginal_Xs = marginal_prob(X)
    marginal_Ys = marginal_prob(Y)
    joint_XYs = joint_prob(X, Y)
    mi = sum([joint_XYs[x, y] * np.log(
        joint_XYs[x, y] /
        (marginal_Xs[x]*marginal_Ys[y]))
        for x in X.unique()
        for y in Y.unique()
        if joint_XYs[x, y] != 0])
    return mi


def conditional_mutual_information(X, Y, Z):
    marginal_Zs = marginal_prob(Z)
    joint_XZs = joint_prob(X, Z)
    joint_YZs = joint_prob(Y, Z)
    joint_XYZs = joint_prob(X, Y, Z)
    mi = sum([joint_XYZs[x, y, z] * np.log(
        marginal_Zs[z]*joint_XYZs[x, y, z] /
        (joint_XZs[x, z]*joint_YZs[y, z]))
        for x in X.unique()
        for y in Y.unique()
        for z in Z.unique()
        if joint_XYZs[x, y, z] != 0])

    return mi


def gamma(shape, rate):
    # numpy uses (shape, scale), paper uses (shape, rate)
    scale = 1.0 / rate
    return np.random.gamma(shape, scale)


def digamma(z):
    return sp.special.digamma(z)


@functools.lru_cache(maxsize=None)
def stirling(n, k):
    """

    Unsigned Sitrling numbers of the first kind

    """
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    # Convert to int to avoid float overflow
    n = int(n)
    k = int(k)
    return (n-1)*stirling(n-1, k) + stirling(n-1, k-1)

# TODO maximum recursion depth
# Still probably overflows for large datasets
#   Could maybe choose a base based on the size of the dataset
#   Would have to implement own loggamma


@functools.lru_cache(maxsize=None)
def log_stirling(n, k):
    if n == k:
        return 0
    if n <= 1 or k == 0:
        return np.NINF
    if k == 1:
        return loggamma(n)

    lns_prev = log_stirling(n-1, k-1)
    lns_up = log_stirling(n-1, k)
    temp = np.log(n-1) + lns_up - lns_prev
    return lns_prev + np.log1p(np.exp(temp))


def beta(a, b):
    return np.random.beta(a, b)


def multinomial(p, n=1):
    return np.argmax(np.random.multinomial(n, p))


@functools.lru_cache(maxsize=None)
def rising_factorial(z, m):
    return int(sp.special.poch(z, m))


@functools.lru_cache(maxsize=None)
def log_rising_factorial(m, n):
    return loggamma(m+n) - loggamma(m)


def normalize_in_log_domain(logs):
    return logs - sum_in_log_domain(logs)


def sum_in_log_domain(logs):
    max_idx = np.argmax(logs)
    max_log = logs[max_idx]

    # Calculate sum of exponent of differences
    sum = 0
    for i in range(len(logs)):
        if i == max_idx:
            sum += 1
        else:
            sum += np.exp(logs[i] - max_log)

    return max_log + np.log(sum)
