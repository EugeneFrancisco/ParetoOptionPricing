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
import copy
import os

BATCH_SIZE = 1024
NUM_EXPERIENCES = 4096
NUM_ITERATIONS = 1600
GAMMA = 1/1.005
NUM_TASKS = 36
SAVE_FREQUENCY = 100
TARGET_UPDATE_FREQUENCY = 50
SAVE_PATH = "results"
RELOAD_PATH = "results/K30H40_trial1/target_1.pth"
ALPHA = 2.0
SCALE = 0.01
TAU = 0.01
MIN_REWARD = -1e8

if torch.backends.mps.is_available():
    train_device = torch.device("mps")
else:
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_device = torch.device("cpu")

def soft_update(target_model, source_model, tau):
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

def make_experience_trace(
        config: Mapping, 
        distribution: Distribution, 
        QFunctionApprox, 
        target_QFunctionApprox, 
        fixed_K = True
        ) -> Iterable[tuple]:
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

        if start_price == 0:
            break

        x = featurize(
            time = start_time, 
            price = start_price,
            strike_price = config['strike_price'] if not fixed_K else None,
            )
        q_values = QFunctionApprox.forward(x)

        action = choose_epsilon_greedy(q_values, epsilon)
        target = torch.zeros((1, 2))

        if start_time == 0:
            raise ValueError("start_time should never be 0")
            break
            if action == 0:
                reward = 0
            else:
                reward = max(start_price, 0) - strike_price

            # min_clipping for stability
            reward = max(reward, MIN_REWARD)
            target[0][action] = reward
            X.append(x)
            targets.append(target)
            break

        if action == 1:
            reward = start_price - strike_price
            target[0][0] = reward
            X.append(x)
            targets.append(target)
            break

        # Now we must have chosen to hold
        new_time = start_time - timeGap
        price_change = start_price*(driftRate * timeGap + (distribution.sample() - mean) * timeGap)
        new_price = price_change + start_price

        if new_time == 0:
            # we just enforce the payout, assuming they act optimally
            reward = max(0, new_price - strike_price)
            target[0][0] = reward
            X.append(x)
            targets.append(target)
            break
        
        # otherwise, the next_state is not terminal

        if new_price <= 0:
            # stocks can't be negative

            target[0][0] = 0
            X.append(x)
            targets.append(target)
            break

        next_state_features = featurize(
            new_time,
            new_price,
            strike_price = config['strike_price'] if not fixed_K else None
        )

        if target_QFunctionApprox is not None:
            next_state_q_values = target_QFunctionApprox.forward(next_state_features)
        else:
            next_state_q_values = QFunctionApprox.forward(next_state_features)

        next_action = choose_greedy(next_state_q_values)
        target[0][0] = GAMMA * next_state_q_values[0][next_action].detach()

        X.append(x)
        targets.append(target)


        start_time = new_time
        start_price = new_price

    return X, targets

def make_data_loader(
        config, 
        distribution, 
        QFunctionApprox, 
        target_QFunctionApprox, 
        iteration,
        epsilon,
        fixed_K = True
        ):
    '''
    Makes a data loader by sampling experience traces for the given distribution and QFunctionApprox.
    iteration is used for any schedulers.
    '''
    
    time_upper_bound = [21, 21, 21, 21]
    time_lower_bound = [1, 1, 1, 1]

    schedule = iteration/NUM_ITERATIONS

    if schedule < 0.25:
        up = time_upper_bound[0]
        low = time_lower_bound[0]
    elif schedule < 0.5:
        up = time_upper_bound[1]
        low = time_lower_bound[1]
    elif schedule < 0.75:
        up = time_upper_bound[2]
        low = time_lower_bound[2]
    else:
        up = time_upper_bound[3]
        low = time_lower_bound[3]

    if iteration % TARGET_UPDATE_FREQUENCY == 0 and target_QFunctionApprox is not None:
        # Copy over target dict once in a while
        target_QFunctionApprox.model.load_state_dict(QFunctionApprox.model.state_dict())


    X_list = [] # an n by d array of states that we've seen, where n is the number of states and d is the feature dimension of the states.

    # an n by 2 array of targets for each state in X, where n is the number of states and 2 is the number of actions.
    # Each row corresponds to one state's target Q values; the action is implied by the column index. 
    
    targets_list = [] 

    num_examples = 0
    QFunctionApprox.model.eval()
    QFunctionApprox.model.to(default_device)

    if target_QFunctionApprox is not None:
        target_QFunctionApprox.model.eval()
        target_QFunctionApprox.model.to(default_device)

    with tqdm(total=NUM_EXPERIENCES, desc="Collecting Experience", unit="samples", leave=False) as pbar:
        while num_examples < NUM_EXPERIENCES:
            
            moneyNess = np.random.uniform(0.3, 1.5, size=NUM_TASKS)
            if schedule < 0.5:
                start_prices = np.random.uniform(0, 41, size=NUM_TASKS)
            if schedule >= 0.5:
                start_prices = np.random.uniform(10, 41, size=NUM_TASKS)

            tasks = [{
                'start_price': start_prices[_],
                'start_time': np.random.randint(low, up),
                'strike_price': start_prices[_]/moneyNess[_] if not fixed_K else config['strike_price'],
                'driftRate': 0.005,
                'timeGap': 1,
                'epsilon': epsilon
            } for _ in range(NUM_TASKS)]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(
                    make_experience_trace, 
                    sampledConfig, 
                    distribution, 
                    QFunctionApprox,
                    target_QFunctionApprox, 
                    fixed_K
                    ) for sampledConfig in tasks]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            for XBatch, targetsBatch in results:
                X_list.extend(XBatch)
                targets_list.extend(targetsBatch)
                num_in_this_experience = len(XBatch)
                num_examples += num_in_this_experience
                pbar.update(num_in_this_experience)

    QFunctionApprox.model.to(train_device)

    if target_QFunctionApprox is not None:
        target_QFunctionApprox.model.to(train_device)

    X = torch.cat(X_list, dim=0).to(train_device)
    targets = torch.cat(targets_list, dim=0).to(train_device)
    dataset = TensorDataset(X, targets)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    return data_loader
    

def q_learning(config, QFunctionApprox, target_QFunctionApprox, fixed_K = True):

    distribution = NegatedParetoDistribution(scale = SCALE, alpha = ALPHA)
    eps_scheduler = epsilon_scheduler(0.1, 0.995, 0.01)

    for i in tqdm(range(NUM_ITERATIONS), desc = "Training Progress"):

        epsilon = next(eps_scheduler)
        data_loader = make_data_loader(
            config, 
            distribution, 
            QFunctionApprox, 
            target_QFunctionApprox, 
            i, 
            epsilon, 
            fixed_K)

        total_loss = 0
        count = 0

        progress_bar = tqdm(data_loader, desc="Training Progress", leave=False)

        for x, target in progress_bar:
            count += 1
            loss = QFunctionApprox.backward(x, target)
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        if i % 5 == 0:
            print(f"Iteration {i}: Loss = {total_loss / count}")
        
        #soft_update(target_QFunctionApprox.model, QFunctionApprox.model, TAU)

        if i % SAVE_FREQUENCY == 0:
            model_name = f"model_{i}.pth"
            path_name = os.path.join(SAVE_PATH, model_name)
            torch.save(QFunctionApprox.model.state_dict(), path_name)
            print("model saved")
        
        if i % TARGET_UPDATE_FREQUENCY == 1:
            model_name = f"target_{i}.pth"
            path_name = os.path.join(SAVE_PATH, model_name)
            torch.save(QFunctionApprox.model.state_dict(), path_name)
            print("mtarget saved")
    
    path_name = os.path.join(SAVE_PATH, "model.pth")
    torch.save(QFunctionApprox.model.state_dict(), path_name)
    print("model saved")

ReLoad = False
QFunctionApprox = SimpleNNApprox(learning_rate=0.01, fixed_K = False, fixed_alpha = True)

if ReLoad:
    QFunctionApprox.model.load_state_dict(torch.load(RELOAD_PATH))

target_QFunctionApprox = copy.deepcopy(QFunctionApprox)
target_QFunctionApprox.model.eval()
target_QFunctionApprox.model.to(default_device)

config = {
    'strike_price': 15.0
}

fixed_K = True

q_learning(config, QFunctionApprox, target_QFunctionApprox, fixed_K = fixed_K)


    