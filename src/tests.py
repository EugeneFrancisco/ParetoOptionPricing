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

GAMMA = 1/1.005


QFunctionApprox = SimpleNNApprox(learning_rate=0.01)
QFunctionApprox.model.load_state_dict(torch.load("results/model.pth"))
QFunctionApprox.model.eval()

for share_price in range(20):
    thisState = state(share_price, 2)
    featurized_state = featurize(thisState)
    q_values = QFunctionApprox.model(featurized_state.to(device))
    print(q_values[0][1])