from linalg_utils import is_symmetric, is_positive_defined, calculate_discrepancy
import numpy as np

class GaussSeidelMethod:
    def __init__(self):
        pass
    
    
    def make_symmetric(self, A, b):
        return np.dot(A.T, A), np.dot(A.T, b)
    
    
    def is_converge(self, A):
        return is_symmetric(A) and is_positive_defined(A)
    
    
    def solve(self, A, b, x0=np.zeros_like(b), eps=1e-5, max_it=1e5, full_print=False):
        B, g = self.make_symmetric(A, b)
        difference = 1
        it = 0
        x_k = np.copy(x0)
        x_k_1 = np.copy(x0)
        while(difference > eps and it < max_it):
            x_k = np.copy(x_k_1)
            for i, _ in enumerate(x0):
                accumulator = 0
                accumulator += -np.dot(B[i, :i], x_k_1[:i]) - np.dot(B[i, i+1:], x_k[i+1:]) + g[i]
                x_k_1[i] = accumulator/B[i][i]
                it += 1
            difference = np.linalg.norm(x_k-x_k_1)
        discrepancy = calculate_discrepancy(A, b, x_k)
        is_system_converge = "Matrix converges" if self.is_converge(B) else "Matrix doesn't converge"
        if full_print:
            self.full_print(difference, it, x_k, discrepancy, is_system_converge)
        return x_k
    
    
    def full_print(self, difference, it, x_k, discrepancy, is_system_converge):
        print(is_system_converge)
        print("Error is: {0:.5E}\nNumber of iterations: {1}".format(difference, it))
        solution_str = ", ".join(["x[{0}] = {1:.5f}".format(i, x_i)
                                 for i, x_i in enumerate(x_k)])
        discrepancy_str = ", ".join(["eps[{0}] = {1:.5E}".format(i, eps_i)
                                    for i, eps_i in enumerate(discrepancy)])
        print("Solution is:\n" + solution_str)
        print("Discrepancy is:\n" + discrepancy_str)
