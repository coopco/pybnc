import functools
import scipy as sp
from scipy.special import loggamma, logsumexp

import numpy as np


def marginal_prob(X):
    """Returns marginal probability distribution of X"""
    marginals = {x: len(X[X == x]) / len(X) for x in X.unique()}
    return marginals


# TODO n-ary functions
def joint_prob_2(X, Y):
    """Returns joint probability distribution of X and Y"""
    joints = {(x, y): len(X[(X == x) & (Y == y)]) / len(X)
              for x in X.unique()
              for y in Y.unique()}
    return joints


def joint_prob_3(X, Y, Z):
    """Returns joint probability distribution of X, Y, and Z"""
    joints = {(x, y, z): len(X[(X == x) & (Y == y) & (Z == z)]) / len(X)
              for x in X.unique()
              for y in Y.unique()
              for z in Z.unique()}
    return joints


def joint_prob(*Xs):
    """Returns joint probability distribution of arguments"""
    if len(Xs) == 2:
        return joint_prob_2(*Xs)
    elif len(Xs) == 3:
        return joint_prob_3(*Xs)
    else:
        raise NotImplementedError


def mutual_information(X, Y):
    """Computes MI(X, Y)"""
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
    """Computes MI(X, Y; Z)"""
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
def rising_factorial(z, m):
    """To compute (z+m)! / z!"""
    return int(sp.special.poch(z, m))


@functools.lru_cache(maxsize=None)
def stirling(n, k):
    """Unsigned Sitrling numbers of the first kind."""
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    # Convert to int to avoid float overflow
    n = int(n)
    k = int(k)
    return (n-1)*stirling(n-1, k) + stirling(n-1, k-1)


@functools.lru_cache(maxsize=None)
def log_stirling(n, k):
    """
    Log of unsigned Sitrling numbers of the first kind.
    TODO maximum recursion depth.
    Still might overflow for large datasets.
    """
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


@functools.lru_cache(maxsize=None)
def log_rising_factorial(m, n):
    """Computes log of the rising factorial"""
    return loggamma(m+n) - loggamma(m)


def beta(a, b):
    return np.random.beta(a, b)


def multinomial(p, n=1):
    return np.argmax(np.random.multinomial(n, p))


def normalize_log_space(logs):
    return logs - logsumexp(logs)
