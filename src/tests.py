from utils import produce_trace, step_minibatch
from distributions import Distribution, NegatedParetoDistribution
from typing import Callable, Iterable, Mapping
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
    
config = {
    'start_price': 100.0,
    'driftRate': 0.05,
    'timeGap': 0.01
}

distribution = NegatedParetoDistribution(scale= config["timeGap"]**0.5, alpha=1.5)
next_prices = step_minibatch(config = config, distribution = distribution, batch_size=1000)
plt.hist(next_prices, bins=100, density=True)
plt.xlabel('Next Price')
plt.show()