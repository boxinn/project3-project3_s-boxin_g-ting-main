"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
alphas  : nx1 vector or alpha values
C : regularization constant

Output:
bias : the scalar hyperplane bias of the kernel SVM specified by alphas

Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
0<alpha<C
"""

import numpy as np

def recoverBias(K,yTr,alphas,C):
    bias = 0
    
    # YOUR CODE HERE
    d,n=np.shape(yTr)
    al=alphas
    for i in range(0,n):
        if al[i]<1e-5 and al[i]>C:
            al[i]=0

    a=n-np.count_nonzero(al)


    sum2=0
    for i in range(0,n):
        if al[i]!=0:
            for j in range(0,n):
                sum1=al[i]*yTr[i]*K[i,j]
        sum2=sum2+yTr[i]-sum1
    bias = sum2/a


    
    return bias 
    
