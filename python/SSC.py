"""
Sparse Subspace Clustering
ECE 510 Final Project

Winter 2020
"""
import numpy as np
import scipy as sp
import math
from numpy import linalg as lin

def Tau(v, eta):
    r"""
    Shrinkage-thresholding operator
    Inputs:
    -------
        v: Matrix size N x N
        eta: threshold

    Outputs:
    -------
        result: Matrix size N x N
    """
    return np.maximum(0,np.abs(v) - (eta * np.ones_like(v))) * np.sign(v)

def SSC(Y,rho,alpha_z,alpha_e,maxIter,eps=2e-4):
    r"""

    Inputs:
    -------
        Y: data matrix of size D x N
        rho: Objective function parameter, value > 0
        alpha_z: Balancing term for Z, value > 1
        alpha_e: Balancing term for E, value > 1
        maxIter: Total number of iterations
        eps: Error tolerance, 2e-4 recommended
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
    yty = np.matmul(Y.T,Y)
    yty = np.abs(yty - np.diag(np.diag(yty)))
    lam_z = alpha_z/np.min(np.max(yty))
    lam_e = alpha_e/np.min(lin.norm(Y,ord=1,axis=1)) #as per John's suggestion

    #These are some repetitive computations used inside the loop
    one_one_T = np.ones((N,N))
    one_matrix = np.ones((N,N))
    left = lam_z * (np.matmul(Y.T,Y)) + rho * np.eye(N) + rho * one_one_T
    yty = np.matmul(Y.T,Y)
    for kk in range(maxIter):
        E_prev = E
        A_prev = A
        #Updating A using CG solver
        #right =  lam_z * np.matmul(Y.T,(Y - E)) + rho * (one_one_T + C) - np.matmul(np.ones((N,1)),delta.T) - Delta
        right = lam_z*yty + rho * (C - Delta*(1/rho)) + rho * (np.matmul(np.ones((N,1)),(np.ones((1,N)) - delta.T*(1/rho))))
        for  i in range(len(right)):
            A[:,i] = sp.sparse.linalg.cgs(left,right[:,i])[0]

        #Updating C
        v = A + (Delta * (1/rho))
        J = Tau(v, 1/rho)
        C = J - np.diag(np.diag(J))

        #Updating E
        v = Y - np.matmul(Y,A)
        E = Tau(v,lam_e/lam_z)

        #Updating delta
        delta = delta + rho * (np.matmul(A.T,np.ones((N,1))) - 1)

        #Updating Delta
        Delta = Delta + rho * (A - C)

        #error check
        if lin.norm((np.matmul(A.T,one_matrix) - one_matrix), np.inf) <= eps\
            and lin.norm(A-C,np.inf) <= eps and lin.norm(A-A_prev,np.inf)<= eps\
            and lin.norm(E-E_prev,np.inf) <= eps:
            break

    return C
