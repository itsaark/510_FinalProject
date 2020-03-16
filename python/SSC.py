"""
Sparse Subspace Clustering
ECE 510 Final Project

Spring 2019
"""
import numpy as np
import scipy as sp
import math
from numpy import linalg as lin

def SSC(Y,rho,lam,maxIter):
    r"""


    Inputs:
    -------
        Y: data matrix of size D x N
        rho:
        lam:
        maxIter:

    Outputs:
    -------
        C: Optimal sparse coefficient matrix of size N x N
    """
    D = Y.shape[0]
    N = Y.shape[1]
    C = np.zeros((N,N))
    A = np.zeros((N,N))
    Delta = np.zeros((N,N))
    delta = np.zeros((N,1))
    E = np.zeros((D,N))
