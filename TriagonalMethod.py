import numpy as np
from linalg_utils import *

class TriagonalMethod:
    def solve(self, A, b, is_print=False):
        self.alpha = np.zeros_like(b)
        self.beta = np.zeros_like(b)
        self.a, self.c, self.b, self.f, self.A = self.make_triagonal(np.copy(A), np.copy(b))
        self.x = np.zeros_like(b)
        self.size = len(b) - 1
        middle = int(np.ceil(self.size/2))
        self.solve_upper_part(middle)
        self.solve_lower_part(middle)
        self.substitute(middle)
        discrepancy = calculate_discrepancy(A, b, self.x)
        if is_print:
            matrix_str = ""
            for row, b_i in zip(self.A, self.f):
                matrix_str += "||"
                matrix_str += " ".join(["{0:10.5f}".format(x_i) for x_i in row])
                matrix_str += "| {0:10.5f}||\n".format(b_i)
            solution_str = "\n".join(["x[{0}] = {1:10.5f}".format(i, x_i)
                                 for i, x_i in enumerate(x)])
            discrepancy_str = "\n".join(["eps[{0}] = {1:14.5e}".format(i, eps_i)
                                    for i, eps_i in enumerate(discrepancy)])
            print("Augmented Matrix is: \n" + matrix_str)
            print("Solution is:\n" + solution_str)
            print("Discrepancy is:\n" + discrepancy_str)
        return self.x
        
    def solve_upper_part(self, middle):
        self.alpha[0] = self.b[0]/self.c[0]
        self.beta[0] = self.f[0]/self.c[0]
        for i in range(1, middle+1):
            denominator = (self.c[i] - self.a[i-1]*self.alpha[i-1])
            self.alpha[i] = self.b[i]/denominator
            self.beta[i] = (self.f[i] + self.a[i-1]*self.beta[i-1])/denominator
    
    def solve_lower_part(self, middle):
        n = self.size
        self.alpha[n] = self.a[n-1]/self.c[n]
        self.beta[n] = self.f[n]/self.c[n]
        for i in range(n-1, middle, -1):
            denominator = (self.c[i] - self.b[i]*self.alpha[i+1])
            self.alpha[i] = self.a[i-1]/denominator
            self.beta[i] = (self.f[i] + self.b[i]*self.beta[i+1])/denominator
    
    def make_triagonal(self, A, b):
        self.top_to_down_eliminate(A, b, bias=1)
        self.bottom_to_top_eliminate(A, b, bias=-1)
        return -A.diagonal(-1), A.diagonal(), -A.diagonal(1), b, A
    
    def bottom_to_top_eliminate(self, A, b, bias=0):
        nrows, ncols = A.shape
        for i in range(nrows-2+bias, -1, -1):
            if A[nrows-1+bias, nrows-1] != 0:
                factor = A[i, nrows-1] / A[nrows-1+bias, nrows-1]
                A[i] = factor*A[nrows-1+bias] - A[i]
                b[i] = factor*b[nrows-1+bias] - b[i]
            else:
                factor = A[i, nrows-1] / A[nrows-1, nrows-1]
                A[i] = factor*A[nrows-1] - A[i]
                b[i] = factor*b[nrows-1] - b[i]
        if nrows is not 2:
            self.bottom_to_top_eliminate(A[:nrows-1,:nrows-1], b[:nrows-1], bias)
        
    def top_to_down_eliminate(self, A, b, bias=0):
        nrows, ncols = A.shape
        in_case_of_zero_bias = bias
        for i in range(bias+1, nrows):
            if A[bias, 0] != 0:
                factor = A[i, 0] / A[bias, 0]
                A[i] = factor*A[bias] - A[i]
                b[i] = factor*b[bias] - b[i]
            else:
                factor = A[i, 0] / A[bias-1, 0]
                A[i] = factor*A[bias-1] - A[i]
                b[i] = factor*b[bias-1] - b[i]
        if nrows is not 2:
            self.top_to_down_eliminate(A[1:, 1:], b[1:], bias)
            
    def substitute(self, middle): 
        denominator = 1 - self.alpha[middle]*self.alpha[middle+1]
        self.x[middle] = (self.beta[middle] + self.alpha[middle]*self.beta[middle+1])/denominator
        for i in range(middle-1, -1, -1):
            self.x[i] = self.alpha[i]*self.x[i+1] + self.beta[i]
        for i in range(middle+1, self.size+1):
            self.x[i] = self.alpha[i]*self.x[i-1] + self.beta[i]
        