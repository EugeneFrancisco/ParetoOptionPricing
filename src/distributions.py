from typing import Iterable, Mapping, Callable
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Distribution:
    '''
    A class representing a distribution with a callable PDF.
    '''
    def __init__(self, mean):
        self.mean = mean

    @abstractmethod
    def sample(self) -> float:
        '''
        Returns a random sample from the distribution.
        '''
        pass
    
    @abstractmethod
    def sample_n(self, n) -> np.ndarray:
        '''
        Returns an np.ndarray of n random samples from the distribution.
        This is useful for generating a batch of samples.
        args:
            n: The number of samples to generate.
        returns:
            An np.ndarray of shape (n,) containing n random samples from the distribution.
        ''' 

        pass

    def __call__(self) -> float:
        return self.sample()

class NegatedParetoDistribution(Distribution):
    '''
    A class representing a Pareto distribution with a callable PDF.
    '''
    def __init__(self, scale: float, alpha: float):
        self.scale = scale
        self.alpha = alpha
        mean = -scale * alpha / (alpha - 1) if alpha > 1 else np.inf
        super().__init__(mean)

    def sample(self) -> float:
        return -(np.random.pareto(self.alpha) + 1) * self.scale
    
    def sample_n(self, n: int) -> np.ndarray:
        '''
        Returns an np.ndarray of n random samples from the distribution.
        This is useful for generating a batch of samples.
        args:
            n: The number of samples to generate.
        returns:
            An np.ndarray of shape (n,) containing n random samples from the distribution.
        '''
        return -(np.random.pareto(self.alpha, size=n) + 1) * self.scale

class NormalDistribution(Distribution):
    '''
    A class representing a normal distribution with a callable PDF.
    '''
    def __init__(self, mean: float, stddev: float):
        super().__init__(mean)
        self.stddev = stddev

    def sample(self) -> float:
        return np.random.normal(self.mean, self.stddev)