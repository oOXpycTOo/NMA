def back_substitution(A, b, start_from="bottom"):
    if(start_from == "top"):
        return substitute_from_top(A, b)
    elif(start_from == "bottom"):
        return substitute_from_bottom(A, b)

def substitute_from_top(A, b):
    x = np.zeros_like(b)
    for i in range(0, A.shape[0]):
        x[i] = (b[i] - np.dot(A[i, 0:i], x[0:i])) / A[i][i]
    return x

def substitute_from_bottom(A, b):
    x = np.zeros_like(b)
    n = A.shape[0] - 1
    zero = -1
    for i in range(n, zero, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i,i]
    return x
