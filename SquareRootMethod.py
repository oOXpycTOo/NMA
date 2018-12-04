from linalg_utils import back_substitution
import numpy as np

def cholessky_decomposition(A):
    S = np.zeros_like(A)
    S[0,0] = np.sqrt(A[0,0])
    S[0,1:] = A[0,1:]/S[0,0]
    for i in range(1, S.shape[0]):
        S[i, i] = np.sqrt(np.abs(A[i, i] - np.sum(S[0:i, i]**2)))
        for j in range(i+1, S.shape[1]):
            S[i][j] = (A[i][j] - np.dot(S[0:i, i], S[0:i, j])) / S[i][i]
    return S

def solve(A, b):
    U = cholessky_decomposition(A)
    y = back_substitution(U.T, b, "top")
    x = back_substitution(U, y, "bottom")
    return x
