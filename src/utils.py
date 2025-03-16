from typing import Iterable, Mapping, float
import numpy as np
import matplotlib.pyplot as plt

def produce_trace(config: Mapping)-> Iterable[float]:
    '''
    Produces a trace of security prices based on the given configuration.
    args:
        config: A dictionary containing the configuration for the trace. Must have the keys
            'start_price' (float) and 'alpha' (float), where alpha is the default parameter
            the pareto distribution. Optional key 'driftRate' (float) can be provided. 
            'timeGap' (float) is also optional and defaults to 1.0.
    returns:
        A generator that yields a sequence of security prices.
    '''

    start_price = config['start_price']
    alpha = config['alpha']
    driftRate = config.get('driftRate', 0.0)
    timeGap = config.get('timeGap', 1.0)

    if alpha <= 0:
        raise ValueError('alpha must be greater than 0')
    
    yield start_price

    while True:
        price_change = start_price*(driftRate * timeGap + np.random.pareto(alpha) * timeGap)
        new_price = price_change + start_price
        yield new_price
        start_price = new_price
    

if __name__ == '__main__':
    # Test the produce_trace function
    config = {
        'start_price': 100.0,
        'alpha': 2.0,
        'driftRate': 0.01,
        'timeGap': 1.0
    }

    trace = produce_trace(config)
    prices = [next(trace) for _ in range(100)]
    