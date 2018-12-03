class IterativeMethod:
    def __init__(self, A, b, x0, eps=1e-8):
        self.B = 0
        self.g = 0
        self.x = x0
        self.canonize(A, b)
        self.eps = eps
        
    def canonize(self, A, b):
        self.B = np.eye(*A.shape) - np.dot(A.T, A)/np.linalg.norm(np.dot(A.T, A))
        self.g = np.dot(A.T, b) / np.linalg.norm(np.dot(A.T, A))
        

    def solve(self):
        difference = 1
        it = 0
        x_prev = float('inf')
        while(difference >= self.eps and it < 1000):
            x_prev = np.copy(self.x)
            self.x = np.dot(self.B, self.x) + self.g
            difference = np.linalg.norm(self.x - x_prev)
            print("="*50)
            print("Accuracy is {0}\nIteration is {1}".format(difference, it))
            it += 1
        return x_prev
