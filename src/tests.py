from utils import produce_trace, step_minibatch, choose_epsilon_greedy, choose_greedy, featurize
from distributions import Distribution, NegatedParetoDistribution
from function_approx import SimpleNNApprox, DummyQApprox
from typing import Callable, Iterable, Mapping
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from state import state

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_optimal_value(QFunctionApprox: SimpleNNApprox, time: int, price: float, strike_price: float) -> float:
    current_state_featurized = featurize(time = float(time), price = float(price)).to(device)
    q_values = QFunctionApprox.forward(current_state_featurized)
    optimal_value = torch.max(q_values).item()
    return optimal_value

#PATH_NAME = "results/K20H160_trial1/model_100.pth"
PATH_NAME = "results/K10H40_SUCCESS/model.pth"
STRIKE_PRICE = 30
START_TIME = 1
TIME_STEPS = 21
SHARE_PRICES = 30

QFunctionApprox10 = SimpleNNApprox(learning_rate=0.01)
QFunctionApprox10.model.load_state_dict(torch.load(PATH_NAME))
QFunctionApprox10.model.eval()

QFunctionApprox15 = SimpleNNApprox(learning_rate=0.01)
QFunctionApprox15.model.load_state_dict(torch.load("results/K15H40_SUCCESS/model.pth"))
QFunctionApprox15.model.eval()

QFunctionApprox20 = SimpleNNApprox(learning_rate=0.01)
QFunctionApprox20.model.load_state_dict(torch.load("results/K20H40_SUCCESS/model.pth"))
QFunctionApprox20.model.eval()

QFunctionApprox25 = SimpleNNApprox(learning_rate=0.01)
QFunctionApprox25.model.load_state_dict(torch.load("results/K25H40_trial1/model.pth"))
QFunctionApprox25.model.eval()

QFunctionApprox30 = SimpleNNApprox(learning_rate=0.01)
QFunctionApprox30.model.load_state_dict(torch.load("results/K30H40_SUCCESS/model.pth"))
QFunctionApprox30.model.eval()

QFunctionApprox35 = SimpleNNApprox(learning_rate=0.01)
QFunctionApprox35.model.load_state_dict(torch.load("results/K35H40_SUCCESS/model.pth"))
QFunctionApprox35.model.eval()

models = [QFunctionApprox10, QFunctionApprox15, QFunctionApprox20, QFunctionApprox25, QFunctionApprox30, QFunctionApprox35]
strikes = [10, 15, 20, 25, 30, 35]



def plot_value_function_of_k():

    for TIME_TILL_EXPIRY in [16, 20]:

        for query_price in range(11, 41, 5):
            option_prices = []
            print("query price: ", query_price, end="\n\n")
            for i, strike in enumerate(strikes):
                optimal_value = get_optimal_value(models[i], time = TIME_TILL_EXPIRY, price = query_price, strike_price = strike)
                option_prices.append(optimal_value)
                print(f"({strike}, {round(optimal_value, 2)})")
    # plt.plot(strikes, option_prices)

# plt.xlabel("Strike Price")
# plt.ylabel("Option Price")
# plt.title("Option Price vs Strike Price")
# plt.show()

# assymmetry test

def assymetry_test():

    for i, strike in enumerate(strikes):
        price = strike * 1.5
        optimal_value = get_optimal_value(models[i], time = 20, price = price, strike_price = strike)
        print(round(optimal_value, 2))


#plotting the optimal value as a function of time

def plot_optimal_function_of_time():

    for time in range(1, TIME_STEPS):
        current_state_featurized = featurize(time = float(time), price = float(STRIKE_PRICE)).to(device)
        q_values = QFunctionApprox30.forward(current_state_featurized)
        optimal_value = torch.max(q_values).item()
        print(f"({time}, {round(optimal_value, 2)})")
    
plot_value_function_of_k()


# optimal_values = np.zeros((TIME_STEPS - START_TIME, SHARE_PRICES))
# QFunctionApprox.model.to(device)

# for t in range(START_TIME, TIME_STEPS):
#     for p in range(1, SHARE_PRICES):
#         current_state_featurized = featurize(time = float(t), price = float(p)).to(device)
#         q_values = QFunctionApprox.forward(current_state_featurized)
        
#         optimal_value = torch.max(q_values).item()
#         # print(f"({t}, {p}, {round(optimal_value, 2)})")
#         optimal_values[t - START_TIME, p - 1] = optimal_value
#         #optimal_values[t - START_TIME, p] = max(torch.max(q_values).item(), 0)

# print("\n\n")

# for p in range(1, SHARE_PRICES):
#     current_state_featurized = featurize(time = 1.0, price = float(p)).to(device)
#     q_values = QFunctionApprox.forward(current_state_featurized)
#     print(q_values[0][0].item(), q_values[0][1].item())

# plt.figure(figsize=(10, 6))
# plt.imshow(optimal_values.T, cmap="viridis", origin="lower", aspect="auto")
# plt.colorbar(label="Optimal Value (Max Q-value)")
# plt.ylabel("Share Price")
# plt.xlabel("Time Step")
# plt.title("Optimal Value Heat Map")
# plt.show()

# distribution = NegatedParetoDistribution(scale = 0.01, alpha = 2.0)

# max_prices = []
# min_prices = []
# for trace in range(1000):
#     price_trace = produce_trace(
#         {
#             'start_price': 20,
#             'drift_rate': 0.005,
#             'timeGap': 1.0
#         },
#         distribution = distribution
#     )

#     prices = [next(price_trace) for _ in range(80)]

#     max_prices.append(max(prices))
#     min_prices.append(min(prices))


# print(min(min_prices))
# plt.hist(max_prices, bins = 50)
# plt.hist(min_prices, bins = 50)
# plt.show()



