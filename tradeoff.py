import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state

def make_data(n_samples, n_sets, random_state=None):
    random = check_random_state(random_state)
    x_r = np.sort(random.uniform(-10, 10, (n_samples,)))
    #Function f(x)
    function = np.sin(x_r) * np.exp((-x_r**2)/16)
    y = np.zeros((n_sets, n_samples))
    for index in range(y.shape[0]):
        y[index, :] = function + 0.1*random.normal(0, 1, n_samples)
    return (x_r, y)

if __name__ == "__main__":
    n_samples = 1000
    n_sets = 50
    (x, y) = make_data(n_samples, n_sets)
    print(np.var(y,0))
