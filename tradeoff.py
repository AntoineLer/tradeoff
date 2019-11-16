import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge

def make_data(n_samples, random_state=None):
    x = np.sort(np.random.uniform(-10, 10, size=n_samples))
    epsilon = np.random.normal(0, 1, n_samples)*0.1
    function = np.sin(x) * np.exp((-x**2)/16)
    y = function + epsilon
    true_y = function
    plt.figure()
    plt.plot(x, y)
    plt.savefig("bite")
    return x, y, true_y

if __name__ == "__main__":
    print("Hello world")
    print("tg")
    make_data(1000)
