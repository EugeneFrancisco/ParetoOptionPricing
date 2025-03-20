from utils import choose_epsilon_greedy, featurize, epsilon_scheduler, choose_greedy
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
import concurrent.futures

if torch.backends.mps.is_available():
    train_device = torch.device("mps")
else:
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_device = torch.device("cpu")

GAMMA = 1/1.005

distribution = NegatedParetoDistribution(scale = 0.03, alpha = 2.0)

QFunctionApprox = SimpleNNApprox(learning_rate=0.01)

def make_experience_trace(config: Mapping, distribution: Distribution, QFunctionApprox) -> Iterable[tuple]:
    '''
    Produces a trace of experiences based on the given configuration.
    args:
        config: A dictionary containing the configuration for the trace. Must have the keys
            'start_price' (float), the key 'start_time' (int) which is the time till expiration that this trace begins at,
            and the key 'strike_price' (float).
            Optional key 'driftRate' (float) can be provided. 
            'timeGap' (float) is also optional and defaults to 1.0.
        distribution: A Distribution object that provides the price dynamics. Namely, distribution has
            a sample method which is proportional to the change in price.
            The mean is also stored in the distributin object.
        QFunctionApprox: A function approximation object that provides the current estimate of the
            Q values for each state-action pair. We need this to choose the next action epsilon greedily.
    '''
    start_price = config['start_price']
    start_time = config['start_time']
    strike_price = config['strike_price']
    epsilon = config['epsilon']
    driftRate = config.get('driftRate', 0.0)
    timeGap = config.get('timeGap', 1.0)

    mean = distribution.mean

    X = []
    targets = []

    while True:
        current_state = state(time = start_time, price = start_price)
        q_values = QFunctionApprox(current_state)

        action = choose_epsilon_greedy(q_values, epsilon)

        x = featurize(current_state)
        target = torch.zeros((1, 2))

        if start_time == 0 or action == 1:
            # the terminal state.
            # we have to check if we entered because date expired or because we chose to execute.
            reward = 0 if action == 0 else start_price - strike_price
            target[0][action] = reward
            X.append(x)
            targets.append(target)
            break
        
        price_change = start_price*(driftRate * timeGap + (distribution.sample() - mean) * timeGap)
        new_price = price_change + start_price
        new_time = start_time - timeGap
        new_state = state(time = new_time, price = new_price)

        next_state_features = featurize(new_state)
        next_state_features = next_state_features.to(default_device)
        next_state_q_values = QFunctionApprox.forward(next_state_features)

        next_action = choose_greedy(next_state_q_values)
        target[0][0] = GAMMA * next_state_q_values[0][next_action].detach()

        X.append(x)
        targets.append(target)


        start_time = new_time
        start_price = new_price

    return X, targets

# time_upper_bounds = [20, 30, 40, 50, 60, 70, 80, 90, 100, 100, 100, 100, 100, 100, 100]
# time_lower_bounds = [1, 1, 1, 1, 1, 1, 1, 20, 30, 40, 60, 70, 80, 80, 90]

eps_scheduler = epsilon_scheduler(0.1, 0.995, 0.01)
for i in range(500):

    epsilon = next(eps_scheduler)
    schedule = int(i/10)
    X_list = [] # an n by d array of states that we've seen, where n is the number of states and d is the feature dimension of the states.

    # an n by 2 array of targets for each state in X, where n is the number of states and 2 is the number of actions.
    # Each row corresponds to one state's target Q values; the action is implied by the column index. 
    
    targets_list = [] 
    num_examples = 0
    QFunctionApprox.model.eval()
    QFunctionApprox.model.to(default_device)

    with tqdm(total=3000, desc="Collecting Experience", unit="samples") as pbar:
        while num_examples < 3000:
            tasks = [{
                'start_price': np.random.uniform(1, 21),
                'start_time': np.random.randint(1, 11),
                'strike_price': 10,
                'driftRate': 0.005,
                'timeGap': 1,
                'epsilon': epsilon
            } for _ in range(36)]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(make_experience_trace, config, distribution, QFunctionApprox) for config in tasks]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            for XBatch, targetsBatch in results:
                X_list.extend(XBatch)
                targets_list.extend(targetsBatch)
                num_in_this_experience = len(XBatch)
                num_examples += num_in_this_experience
                pbar.update(num_in_this_experience)

    QFunctionApprox.model.to(train_device)
    X = torch.cat(X_list, dim=0).to(train_device)
    targets = torch.cat(targets_list, dim=0).to(train_device)

    dataset = TensorDataset(X, targets)
    batch_size = 1000

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    total_loss = 0
    count = 0

    progress_bar = tqdm(data_loader, desc="Training Progress")

    for x, target in progress_bar:
        count += 1
        loss = QFunctionApprox.backward(x, target)
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    
    if i % 5 == 0:
        print(f"Iteration {i}: Loss = {total_loss / count}")
    

torch.save(QFunctionApprox.model.state_dict(), 'results/model.pth')