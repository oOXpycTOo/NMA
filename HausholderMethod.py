from linalg_utils import back_substitution
import numpy as np

class HausholderMethod:
    def __init__(self, A, b):
        self.A = np.copy(A)
        self.b = np.copy(b)
        self.n = self.b.shape[0]
        self.iteration = 0
        
    def decompose(self):
        for k in range(0, self.n-1):
            self.iteration = k
            s = self.get_s()
            e = self.get_e()
            alpha = self.calculate_alpha(s)
            omega = self.calculate_omega(alpha, s, e)
            A_ = np.copy(self.A[k:,k+1:])
            b_ = np.copy(self.b[k:])
            self.A[k][k] = alpha
            for i in range(k, self.n):
                self.b[i] = b_[i-k] - 2*omega[i]*np.dot(b_, omega[k:])
                for j in range(k+1, self.n):
                    self.A[i][j] = self.A[i][j] - 2*omega[i]*np.dot(A_[:,j-(k+1)], omega[k:])
            self.A[k+1:,k] = 0
    
    def solve(self):
        self.decompose()
        return back_substitution(self.A, self.b)
            
    def get_s(self):
        s = np.copy(self.A[:,self.iteration])
        s[:self.iteration] = 0
        return s
    
    def get_e(self):
        e = np.zeros(self.n)
        e[self.iteration] = 1
        return e
        
    def calculate_alpha(self, s):
        return np.linalg.norm(s)
    
    def calculate_x(self, alpha, s, e):
        return 1 / (np.sqrt(2*np.dot(s, s - alpha * e)))
    
    def calculate_omega(self, alpha, s, e):
        x = self.calculate_x(alpha, s, e)
        return x * (s - alpha*e)
