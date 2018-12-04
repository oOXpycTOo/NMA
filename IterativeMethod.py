class IterativeMethod:
    def __init__(self):
        pass
        
        
    def canonize(self, A, b):
        B = np.eye(*A.shape) - np.dot(A.T, A)/np.linalg.norm(np.dot(A.T, A))
        g = np.dot(A.T, b) / np.linalg.norm(np.dot(A.T, A))
        return B, g

    
    def is_converge(self, B):
        if np.linalg.norm(B) < 1:
            return "Method converges (||B|| is equal to {})".format(np.linalg.norm(B))
        if np.max(np.abs(np.linalg.eigvalsh(B))):
            return "Method converges (max eigenvalue is: {})".format(np.max(np.abs(np.linalg.eigvalsh(B))))
        return "Method doesn't converge"
    
    
    def estimate_iterations(self, B, g, eps):
        norm_B = np.linalg.norm(B)
        norm_g = np.linalg.norm(g)
        if(norm_B > 1):
            return "Impossible to estimate"
        return (np.log10(eps) + np.log10(1-norm_B) - np.log10(norm_g))/norm_g - 1
        
        
    def solve(self, A, b, x0=np.zeros_like(b), eps=1e-5, max_it=1e8, full_print=False):
        B, g = self.canonize(A, b)
        difference = 1
        it = 0
        x_prev = np.copy(x0)
        x_next = np.copy(x0)
        while(difference >= eps and it < max_it):
            x_next = np.dot(B, x_next) + g
            difference = np.linalg.norm(x_next - x_prev)
            x_prev = np.copy(x_next)
            it += 1
        discrepancy = calculate_discrepancy(A, b, x_prev)
        estimated_iterations = self.estimate_iterations(B, g, eps)
        is_method_converge = self.is_converge(B)
        if full_print:
            self.full_print(difference, it, x_prev, discrepancy, estimated_iterations, is_method_converge)
        return x_prev
    
    def full_print(self, difference, it, x_k, discrepancy, estimated_iterations, is_method_converge):
        print(is_method_converge)
        print("Estimated iterations: {0}\n".format(estimated_iterations))
        print("Error is: {0:.5E}\nNumber of iterations: {1}\n".format(difference, it))
        solution_str = ", ".join(["x[{0}] = {1:.5f}".format(i, x_i)
                                 for i, x_i in enumerate(x_k)])
        discrepancy_str = ", ".join(["eps[{0}] = {1:.5E}".format(i, eps_i)
                                    for i, eps_i in enumerate(discrepancy)])
        print("\nSolution is:\n" + solution_str)
        print("\nDiscrepancy is:\n" + discrepancy_str)
