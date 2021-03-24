import numpy as np
import math as math

dataset = np.array([[1,2,3,4,9],
       [2,4,6,8,10],
       [3,4,4,4,4],
       [3,3,5,2,1]])

def corr(data):
    return np.corrcoef(dataset)

dataset_corr = corr(dataset)

def kmo(dataset_corr):
    corr_inv = np.linalg.inv(dataset_corr)
    nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
    A = np.ones((nrow_inv_corr,ncol_inv_corr))
    for i in range(0,nrow_inv_corr,1):
        for j in range(i,ncol_inv_corr,1):
            A[i,j] = -(corr_inv[i,j])/(math.sqrt(corr_inv[i,i]*corr_inv[j,j]))
            A[j,i] = A[i,j]
    dataset_corr = np.asarray(dataset_corr)
    kmo_num = np.sum(np.square(dataset_corr)) - np.sum(np.square(np.diagonal(A)))
    kmo_denom = kmo_num + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
    kmo_value = kmo_num / kmo_denom
    return kmo_value

kmo(dataset_corr)
print(kmo(dataset_corr))