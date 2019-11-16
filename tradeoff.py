import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.utils import check_random_state

def f(X):
    return np.sin(X) * np.exp((-X**2)/16)

def make_data(n_samples, n_sets, random_state=None):
    random = check_random_state(random_state)
    x_train = random.uniform(-10, 10, (n_sets, n_samples))
    #Function f(x)
    function = f(x_train)
    epsilon = random.normal(0, 1, (n_sets, n_samples))
    y_train = function + 0.1*epsilon

    x_test = np.sort(random.uniform(-10, 10, n_samples))
    function = f(x_test)
    y_test = function + 0.1*random.normal(0, 1, n_samples)
    return (x_train, y_train, x_test, y_test)

def linear_model(x_train, y_train, x_test, y_test, residual_error):
    linear_estimators = [Ridge().fit(x.reshape(-1, 1), y) for x, y in zip(x_train, y_train)]
    linear_prediction = [estimator.predict(x_test.reshape(-1, 1)) for estimator in linear_estimators]

    noise = np.full(x_train.shape[1], residual_error)
    squared_bias = (f(x_test) - np.mean(linear_prediction, 0))
    variance = np.var(linear_prediction, 0)
    tradeoff = noise + squared_bias + variance
    print(variance)
    plt.figure()
    plt.plot(x_test, tradeoff, label = "Expected error")
    plt.plot(x_test, variance, label = "Variance")
    plt.plot(x_test, noise, label ="Residual Error")
    plt.plot(x_test, squared_bias, label ="Squared Bias")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig("tradeoff_Q_d.pdf")
    return

def noise(x, y):
    residual_error = []

    for i in range(x.shape[0]):
        for k in range(x[i].shape[0]):
            residual_error.append((y[i][k] - f(x[i][k]))**2)
    return np.mean(np.array(residual_error))

if __name__ == "__main__":
    n_samples = 1000
    n_sets = 50
    (x_train, y_train, x_test, y_test) = make_data(n_samples, n_sets)
    residual_error = noise(x_train, y_train)

    linear_model(x_train, y_train, x_test, y_test, residual_error)
