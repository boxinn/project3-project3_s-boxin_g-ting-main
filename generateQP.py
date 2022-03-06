"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
C : regularization constant

Output:
Q,p,G,h,A,b as defined in qpsolvers.solve_qp

A call of qpsolvers.solve_qp(Q, p, G, h, A, b) should return the optimal nx1 vector of alphas
of the SVM specified by K, yTr, C. Just make these variables np arrays.

"""
import numpy as np


def generateQP(K, yTr, C):
    yTr = yTr.astype(np.double)
    n = yTr.shape[0]
    
    # YOUR CODE HERE
    G=np.concatenate((np.positive(np.eye(n)),np.negative(np.eye(n))),axis=0)
    h=np.concatenate((np.ones([n,1])*C,np.zeros([n,1])),axis=0)
    A=yTr.T
    b=np.zeros(1)
    p=-1*np.ones(n)
    Q=yTr*K*yTr.T

    return Q, p, G, h, A, b