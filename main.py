import pandas as pd
import matplotlib
import numpy as np
if __name__ == '__main__':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint

file = 'predator-prey-data.csv'
data = pd.read_csv(file, index_col=0)


def lv_system(state, t, a, b, d, g):
    """Lotka-Volterra predator-prey model."""
    x, y = state
    dx = a * x - b * x * y
    dy = d * x * y - g * y
    return [dx, dy]


def sim_lv(params, time):
    """Simulate the Lotka-Volterra model."""
    a, b, d, g = params
    start = [data['x'][0], data['y'][0]]
    simulated = odeint(lv_system, start, time, args=(a, b, d, g))
    return simulated


def sse(params, time, data):
    """Calculate the sum of squared errors."""
    simulated = sim_lv(params, time)
    sse = np.sum((data['x'] - simulated[:, 0]) ** 2 +
                 (data['y'] - simulated[:, 1]) ** 2)
    return sse


def mae(params, time, data):
    """Calculate the mean absolute error."""
    simulated = sim_lv(params, time)
    mae = np.mean(np.abs(data['x']-simulated[:, 0]) +
                  np.abs(data['y'] - simulated[:, 1]))
    return mae


def hill_climbing(obj, params, time, data, i, step):
    """Hill climbing algorithm.

    Args:
        obj (function): Objective function to minimize.
        params (array): Initial parameters.
        time (array): Time points.
        data (DataFrame): Original data.
        i (int): Number of iterations.
        step (float): Step size.

    Returns:
        array: Optimized parameters.

    """
    best_params = params
    optimal = obj(best_params, time, data)

    for _ in range(i):
        option = best_params + np.random.normal(0, step, len(best_params))
        option_score = obj(option, time, data)
        if option_score < optimal:
            optimal, best_params = option_score, option

    return best_params


def simulated_annealing(obj, params, time, data, i, temp, stepsize=0.5):
    """Simulated annealing algorithm.

    Args:

        obj (function): Objective function to minimize.
        params (array): Initial parameters.
        time (array): Time points.
        data (DataFrame): Original data.
        i (int): Number of iterations.
        temp (float): Temperature of simulated annealing algorithm.

    Returns:
        array: Optimized parameters.

    """
    best_params = params
    current = params
    current_score = obj(current, time, data)
    optimal = current_score

    for j in range(i):
        temp = temp / np.log2(j + 2)
        step = np.random.normal(0, stepsize, len(best_params))
        option = current+step
        option_score = obj(option, time, data)

        if option_score < current_score or np.exp((current_score - option_score)/temp) > np.random.rand():
            current, current_score = option, option_score

            if option_score < optimal:
                optimal, best_params = option_score, option

    return best_params


def optimize_parameters(data, method, func, i,  temp, array, step=0.202):
    """Optimize parameters of the Lotka-Volterra model.

    Args:
        data (DataFrame): Original data.
        method (str): Optimization method. Either 'hill_climbing' or 'simulated_annealing'.
        func (str): Objective function. Either 'sse' or 'mae'.
        i (int): Number of iterations.
        temp (float): Temperature of simulated annealing algorithm.
        array (array): Initial parameters.

    Returns:
        array: Optimized parameters.

    """
    initial_params = array
    time_series = data['t'].values

    if func == 'sse':
        obj = sse
    elif func == 'mae':
        obj = mae

    if method == 'hill_climbing':
        return hill_climbing(obj, initial_params, time_series, data, i, step)
    elif method == 'simulated_annealing':
        return simulated_annealing(obj, initial_params, time_series, data, i, temp, step)


def calculate_errors(params, original_data):
    """Calculate errors between simulated and original data.

    Args:
        params (array): Optimized parameters.
        original_data (DataFrame): Original data.

    Returns:
        tuple: SSE and MAE errors.

    """
    time_series = original_data['t'].values

    # Calculate SSE and MAE
    sse_error = sse(params, time_series, original_data)
    mae_error = mae(params, time_series, original_data)

    return sse_error, mae_error


def plot_sim(params, data):
    """Plot simulated and original data.

    Args:
        params (array): Optimized parameters.
        data (DataFrame): Original data.
    """
    time_series = data['t'].values
    simulated = sim_lv(params, time_series)

    plt.figure(figsize=(10, 6))
    plt.plot(data['t'], data['x'], 'b-', label='Actual Prey Population')
    plt.plot(data['t'], data['y'], 'r-', label='Actual Predator Population')
    plt.plot(data['t'], simulated[:, 0], 'b--',
             label='Simulated Prey Population')
    plt.plot(data['t'], simulated[:, 1], 'r--',
             label='Simulated Predator Population')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Comparison of Predator-Prey Population Dynamics')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    i = 0
    starting = np.array([-2, -1.15, -0.4, -0.85])  # based on numerous runs
    while i < 10:
        # choose hill_climbing or simulated_annealing and mae or sse
        optimized_params = optimize_parameters(
            data, 'simulated_annealing', 'sse', 10000, 1, starting)
        print("Optimized Parameters:", optimized_params)
        sse_error, mae_error = calculate_errors(optimized_params, data)
        print("Error (SSE) between simulated and original data:", sse_error)
        print("Error (MAE) between simulated and original data:", mae_error)
        starting = optimized_params
        i += 1
        plot_sim(optimized_params, data)
