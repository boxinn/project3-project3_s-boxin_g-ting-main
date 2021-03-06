"""
function K = computeK(kernel_type, X, Z)
computes a matrix K such that Kij=g(x,z);
for three different function linear, rbf or polynomial.

Input:
kernel_type: either 'linear','poly','rbf'
X: n input vectors of dimension d (dxn);
Z: m input vectors of dimension d (dxn);
kpar: kernel parameter (inverse kernel width gamma in case of RBF, degree in case of polynomial)

OUTPUT:
K : nxm kernel matrix
"""
import numpy as np
from l2distance import l2distance

def computeK(kernel_type, X, Z, kpar):
    assert kernel_type in ['linear', 'poly', 'rbf'], kernel_type + ' is an unrecognized kernel type in computeK'
    
    d, n = X.shape
    dd, m = Z.shape
    assert d == dd, 'First dimension of X and Z must be equal in input to computeK'
    
    K = np.zeros((n,m))
    
    # YOUR CODE HERE
    if kernel_type=='linear':
        for i in range (0,n):
            for j in range(0,m):
                K[i,j]=np.dot(X[:,i].T,Z[:,j])


    if kernel_type=='rbf':
        D=l2distance(X,Z)
        for i in range (0,n):
            for j in range(0,m):
                K[i,j]=np.exp(-kpar*(D[i,j]**2))


    if kernel_type=='poly':
        for i in range(0,n):
            for j in range(0,m):
                K[i,j]=np.power((np.dot(X[:,i].T,Z[:,j])+1),kpar)



    return K
