import numpy as np
from utils import MSE_loss, featurize
import torch
from abc import ABC, abstractmethod
from state import state


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

class SimpleNNApprox:
    def __init__(self, learning_rate: float):
        '''
        Initializes a simple two layer neural network function approximator of the Q function. 
        args:
            learning_rate: the learnign rate used for gradient descent updates
        '''
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Using device:", self.device)
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(6, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 2)
        )
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = MSE_loss

        self.model.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        args:
            x: A torch.Tensor of shape (n, d) representing the batch input of states for which we want to calculate
            all the Q values. d is the dimension of the state space (after featurization).
        returns:
            a torch.Tensor of shape (n, 2) representing the Q values for each state-action pair. Note that
            index 0 corresponds to holding and index 1 corresponds to executing.
        '''
        return self.model(x)
    
    def backward(self, x: torch.Tensor, targets: torch.Tensor):
        '''
        Performs one gradient descent update on the current function approximation and updates
        the internal model.
        args:
            x: A torch.Tensor of shape (n, d) representing the batch input of states for which we want to calculate
            all the Q values. d is the dimension of the state space (after featurization).
            targets: A torch.Tensor of shape (n, 2) representing the target Q values for each state-action pair. Note that
            each row of targets should have only one non-zero entry, corresponding to the action taken.
            The non-zero entry should be the target Q value for the action taken.
        returns:
            the loss computed using the MSE loss function.
        '''
        self.model.train()
        self.optimizer.zero_grad()
        predictions = self.forward(x)
        loss = self.loss_fn(predictions, targets)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def __call__(self, currentState: state) -> np.ndarray:
        '''
        Computes the Q values for the given state.
        args:
            currentState: a state object representing the current state of the system.
        returns:
            a numpy array of shape (2,1) representing the Q values for each action. Note that
            index 0 corresponds to holding and index 1 corresponds to executing'
        '''
        self.model.eval()
        with torch.no_grad():
            x = featurize(currentState)
            q_values = self.forward(x)
            return q_values

class DummyQApprox:
    def __call__(self, currentState: state) -> np.ndarray:
        '''
        Computes the Q values for the given state.
        args:
            currentState: a state object representing the current state of the system.
        returns:
            a numpy array of shape (2,1) representing the Q values for each action. For testing
            we always return that holding is higher quality than executing.
        '''
        return torch.tensor([[1], [0]])




    
