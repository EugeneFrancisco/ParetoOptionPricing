from typing import Iterable, Mapping, Callable
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from distributions import Distribution

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
    timeGap = config.get('timeGap', 0.01)

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

    