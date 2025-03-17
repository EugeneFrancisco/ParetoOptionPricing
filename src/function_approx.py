import numpy as np

class LinearApprox:
    def __init__(self, weights: np.ndarray, learning_rate: float):
        '''
        Initializes the linear approximation with the given parameters.
        args:
            weights: A numpy array of shape (d, 1) representing the parameters of the linear function.
        '''
        self.weights = weights
        self.learning_rate = learning_rate
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Computes the linear approximation of the function for the minibatch x.
        args:
            x: A numpy array of shape (n, d) representing the batch input to the function
        returns:
            A numpy array of shape (n,) representing the output of the function.
        '''
        return x @ self.weights
    
    def step(self, hyp: np.ndarray, x: np.ndarray, labels: np.ndarray):
        '''
        Performs one gradient descent update on the current function approximation, and stores
        the new weights locally.
        args:
            hyp: A numpy array of shape (n, 1) representing the current hypothesis for the minibatch,
                i.e., the output of the forward pass.
            x: A numpy array of shape (n, d) representing the batch input to the function.
                This is the same as the input to the forward pass.
            labels: A numpy array of shape (n, 1) representing the labels for the minibatch.
        returns:
            nothing, just updates the weights of the current linear approximation.
        '''
        self.weights = self.weights - self.learning_rate * x.T @ (hyp - labels) / x.shape[0]


    
