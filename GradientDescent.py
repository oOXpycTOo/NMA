import numpy as np
from linalg_utils import is_positive_defined, is_symmetric, calculate_discrepancy, full_print

class GradientDescent:
    def solve(self, A, b, x0, eps=1e-5, max_it=1e8, is_print=False):
        """Solves system of linear equations using gradient descent method
        r for discrepancy
        A - matrix A (coefficients)
        b - free term
        x0 - initial guess
        eps - disired accuracy"""
        x_next = np.copy(x0)
        x_prev = np.copy(x0)
        r = calculate_discrepancy(A, b, x0)
        difference = 1
        it = 0
        while(difference > eps):
            t = np.dot(r, r)/np.dot(np.dot(A, r), r)
            x_next = x_prev - t*r
            difference = np.linalg.norm(x_next - x_prev)
            x_prev = np.copy(x_next)
            r = calculate_discrepancy(A, b, x_prev)
            it += 1
        if is_print:
            full_print(difference, it, x_prev, r)
        return x_prev
