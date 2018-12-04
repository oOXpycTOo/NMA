import numpy as np

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

def calculate_discrepancy(A, b, x):
    return np.dot(A, x) - b

def is_positive_defined(A):
    return np.all(np.linalg.eigvals(A) > 0)

def is_symmetric(A):
    return np.all(A==A.T)

def full_print(difference, it, x_k, discrepancy, is_system_converge="No information"):
        print(is_system_converge)
        print("Error is: {0:8.5e}\nNumber of iterations: {1}".format(difference, it+1))
        solution_str = "\n".join(["x[{0}] = {1:8.5f}".format(i, x_i)
                                 for i, x_i in enumerate(x_k)])
        discrepancy_str = "\n".join(["eps[{0}] = {1:14.5e}".format(i, eps_i)
                                    for i, eps_i in enumerate(discrepancy)])
        print("Solution is:\n" + solution_str)
        print("Discrepancy is:\n" + discrepancy_str)


