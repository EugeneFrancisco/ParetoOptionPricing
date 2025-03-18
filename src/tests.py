from utils import produce_trace, step_minibatch, make_experience_trace
from distributions import Distribution, NegatedParetoDistribution
from function_approx import SimpleNNApprox, DummyQApprox
from typing import Callable, Iterable, Mapping
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
    
config = {
    'start_price': 100,
    'start_time': 100,
    'strike_price': 100,
    'driftRate': 0.005,
    'timeGap': 1
}

distribution = NegatedParetoDistribution(scale = 0.03, alpha = 2.0)

#path = produce_trace(config = config, distribution = distribution)
#price_list_path = [next(path) for _ in range(100)]
#plt.plot(price_list_path)

QFunctionApprox = DummyQApprox()
experience_trace = make_experience_trace(config, distribution, QFunctionApprox)
price_list = [tup[0].price for tup in experience_trace]
plt.plot(price_list)
plt.show()


# next_prices = step_minibatch(config = config, distribution = distribution, batch_size=100)
# plt.hist(next_prices, bins=100, density=True)
# plt.xlabel('Next Price')