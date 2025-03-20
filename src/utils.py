from typing import Iterable, Mapping, Callable
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from distributions import Distribution
import torch
import math
from state import state

def choose_epsilon_greedy( q_values: torch.Tensor, epsilon: float) -> int:
    '''
    Chooses an action using epsilon-greedy policy
    args:
        q_values: a torch.Tensor of shape (2,1) representing the Q values for each action.
        epsilon: the probability of choosing a random action.
    returns:
        An integer representing the chosen action. 0 corresponds to holding and 1 corresponds to executing.
        This is also the index at which that action appears within the q_values tensor.
    '''
    if np.random.rand() < epsilon:
        return np.random.randint(0, 2)
    else:
        return torch.argmax(q_values).item()

def choose_greedy(q_values: torch.Tensor) -> int:
    '''
    Chooses an action using greedy policy
    args:
        q_values: a torch.Tensor of shape (2,1) representing the Q values for each action.
    returns:
        An integer representing the chosen action. 0 corresponds to holding and 1 corresponds to executing.
        This is also the index at which that action appears within the q_values tensor.
    '''
    return torch.argmax(q_values).item()

def featurize(currentState: state) -> torch.Tensor:
    '''
    Featurizes the current state into a torch.Tensor.
    args:
        currentState: a state object representing the current state of the system.
    returns:
        a torch.Tensor of shape (1, 2) representing the featurized state. Note that the returned tensor
        is on the same device as the model. The features are:
        [time, price]
    '''

    
    return torch.tensor([
        currentState.time,
        currentState.price
    ]).unsqueeze(0)

def MSE_loss(predictions: torch.Tensor, target: torch.Tensor) -> float:
    '''
    Computes the mean squared error loss for the predicted Q values and the true Q values.
    args:
        predictions: a torch.Tensor of shape (n, alpha) representing the predicted Q values, where n is the
        batch size and alpha is the number of actions.
        target: a torch.Tensor of shape (n, alpha) representing the true Q values. Note that for each row of
        target, only one entry is non-zero, corresponding to the action taken.
        The non-zero entry should be the target Q value for the action taken.
        The rest should be zero.
    returns:
        The mean squared error between the target Q value and the associated prediction value.
    '''

    nonzero_indices = target.nonzero(as_tuple=True)
    pred_q_values = predictions[nonzero_indices]
    target_q_values = target[nonzero_indices]
    loss = torch.mean((pred_q_values - target_q_values) ** 2)
    return loss

def produce_trace(config: Mapping, distribution: Distribution)-> Iterable[float]:
    '''
    Produces a trace of security prices based on the given configuration.
    args:
        config: A dictionary containing the configuration for the trace. Must have the keys
            'start_price' (float). Optional key 'driftRate' (float) can be provided. 
            'timeGap' (float) is also optional and defaults to 1.0.
        distribution: A Distribution object that provides a callable PDF for the security prices
        as well as a mean.
    returns:
        A generator that yields a sequence of security prices. Specifically, we assume that
        dSt/St ~ (driftRate * dt + (X - mean) * dt), where X is a random variable from the
        provided sampler. The mean is 
    '''

    start_price = config['start_price']
    driftRate = config.get('driftRate', 0.0)
    timeGap = config.get('timeGap', 1.0)

    mean = distribution.mean
    
    yield start_price

    while True:
        price_change = start_price*(driftRate * timeGap + (distribution.sample() - mean) * timeGap)
        new_price = price_change + start_price
        yield new_price
        start_price = new_price

def step_minibatch(
        config: Mapping,
        distribution: Distribution,
        batch_size: int
):
    '''
    Produces a minibatch of size batch_size of next security prices based on the given start_price.
    args:
        config: A dictionary containing the configuration for the trace. Must have
            the keys 'start_price' (float). Optional key 'driftRate' (float) can be provided. 
            'timeGap' (float) is also optional and defaults to 0.01.
        distribution: A Distribution object that provides a callable sampler for the security prices and
            a mean.
        batch_size: The size of the minibatch to produce.
    returns:
        A list of security prices of length batch_size.
    '''
    start_price = config['start_price']
    driftRate = config.get('driftRate', 0.0)
    timeGap = config.get('timeGap', 0.01)
    mean = distribution.mean
    new_prices = start_price + start_price*(driftRate*timeGap + (distribution.sample_n(batch_size) - mean) * timeGap)
    return new_prices


def epsilon_scheduler(initial_epsilon: float, decay_rate: float, min_epsilon: float) -> Iterable[float]:
    '''
    Computes the epsilon value for the given epoch using an exponential decay schedule.
    args:
        initial_epsilon: The initial value of epsilon.
        decay_rate: The rate at which epsilon decays.
        min_epsilon: The minimum value of epsilon.
        epoch: The current epoch.
    returns:
        The epsilon value for the given epoch.
    '''
    count = 0
    while True:
        yield max(min_epsilon, initial_epsilon * decay_rate ** count)
        count += 1

    