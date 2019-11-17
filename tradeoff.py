import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.utils import check_random_state


def f(X):
    return np.sin(X) * np.exp((-X**2)/16)


def make_data(n_samples, n_sets, random_state=None):
    random = check_random_state(random_state)
    x_train = random.uniform(-10, 10, (n_sets, n_samples))
    # Function f(x)
    function = f(x_train)
    epsilon = random.normal(0, 1, (n_sets, n_samples))
    y_train = function + 0.1*epsilon

    x_test = np.sort(random.uniform(-10, 10, n_samples))
    function = f(x_test)
    y_test = function + 0.1*random.normal(0, 1, n_samples)
    return (x_train, y_train, x_test, y_test)


def linear_model(x_train, y_train, x_test, residual_error, number_irrelevant_variables, plot=False, alpha=0):
    if alpha == 0:
        linear_estimators = [LinearRegression().fit(add_irrelevant_variables(
            x, number_irrelevant_variables), y) for x, y in zip(x_train, y_train)]
        linear_prediction = [estimator.predict(add_irrelevant_variables(
            x_test, number_irrelevant_variables)) for estimator in linear_estimators]
    else:
        linear_estimators = [Lasso(alpha=alpha).fit(add_irrelevant_variables(
            x, number_irrelevant_variables), y) for x, y in zip(x_train, y_train)]
        linear_prediction = [estimator.predict(add_irrelevant_variables(
            x_test, number_irrelevant_variables)) for estimator in linear_estimators]

    noise = np.full(x_train.shape[1], residual_error)
    squared_bias = compute_squared_bias(x_test, linear_prediction)
    variance = compute_variance(linear_prediction)
    error = noise + squared_bias + variance

    if plot:
        plt.figure()
        plt.plot(x_test, error, label="Expected error")
        plt.plot(x_test, variance, label="Variance")
        plt.plot(x_test, noise, label="Residual Error")
        plt.plot(x_test, squared_bias, label="Squared Bias")
        plt.xlabel("x")
        plt.ylabel("Error")
        plt.legend()
        plt.savefig("tradeoff_Q_d_linear.pdf")
    return (noise, squared_bias, variance, error)


def non_linear_model(x_train, y_train, x_test, residual_error, number_irrelevant_variables, plot=False, n_neighbors=5):
    non_linear_estimators = [KNeighborsRegressor(n_neighbors=n_neighbors).fit(add_irrelevant_variables(
        x, number_irrelevant_variables), y) for x, y in zip(x_train, y_train)]
    x_test_irrelevante_variable = add_irrelevant_variables(
        x_test, number_irrelevant_variables)
    non_linear_prediction = [estimator.predict(
        x_test_irrelevante_variable) for estimator in non_linear_estimators]

    noise = np.full(x_train.shape[1], residual_error)
    squared_bias = compute_squared_bias(x_test, non_linear_prediction)
    variance = compute_variance(non_linear_prediction)
    error = noise + squared_bias + variance

    """
    Plot results
    """
    if plot:
        plt.figure()
        plt.plot(x_test, error, label="Expected error")
        plt.plot(x_test, variance, label="Variance")
        plt.plot(x_test, noise, label="Residual Error")
        plt.plot(x_test, squared_bias, label="Squared Bias")
        plt.xlabel("x")
        plt.ylabel("Error")
        plt.legend()
        plt.savefig("tradeoff_Q_d_non_linear.pdf")
    return (noise, squared_bias, variance, error)


def compute_variance(predictions):
    return np.var(predictions, 0)


def compute_squared_bias(x, prediction):
    bayes_model = f(x)
    return (bayes_model - np.mean(prediction, 0))**2


def compute_noise(x, y):
    residual_error = []
    for i in range(x.shape[0]):
        bayes_model = f(x[i])
        residual_error.append((y[i] - bayes_model)**2)
    return np.mean(np.array(residual_error))


def add_irrelevant_variables(X, number_irrelevant_variables, random_state=None):
    if number_irrelevant_variables == 0:
        return X.reshape(-1, 1)
    random = check_random_state(random_state)
    variables = np.zeros((X.shape[0], 1 + number_irrelevant_variables))
    for i in range(X.shape[0]):
        variables[i] = np.concatenate(
            ([X[i]], random.uniform(-10, 10, number_irrelevant_variables)))
    return variables.reshape(-1, 1 + number_irrelevant_variables)


def compute_error(n_samples, n_sets, number_irrelevant_variables, alpha=0, n_neighbors=5):
    (x_train, y_train, x_test, y_test) = make_data(n_samples, n_sets)
    residual_error = compute_noise(x_train, y_train)

    """
    Compute linear model errors
    """
    linear_model(x_train, y_train, x_test, residual_error,
                 number_irrelevant_variables, plot=True, alpha=alpha)
    """
    Compute non linear model errors
    """
    non_linear_model(x_train, y_train, x_test, residual_error,
                     number_irrelevant_variables, plot=True, n_neighbors=n_neighbors)


def mean_size_LS(n_samples, n_sets, number_irrelevant_variables, start):
    """
    Variable initialisation
    """
    linear_noise = []
    linear_squared_bias = []
    linear_variance = []
    linear_error = []

    non_linear_noise = []
    non_linear_squared_bias = []
    non_linear_variance = []
    non_linear_error = []

    """
    Compute Mean
    """
    for size in range(start, n_samples):
        if size == 0:
            continue
        (x_train, y_train, x_test, y_test) = make_data(size, n_sets)
        residual_error = compute_noise(x_train, y_train)

        """
        Linear Regression
        """
        (residual_error, squared_bias, variance, error) = linear_model(
            x_train, y_train, x_test, residual_error, number_irrelevant_variables)
        linear_noise.append(np.mean(residual_error))
        linear_squared_bias.append(np.mean(squared_bias))
        linear_variance.append(np.mean(variance))
        linear_error.append(np.mean(error))

        """
        Non Linear Regression
        """
        (residual_error, squared_bias, variance, error) = non_linear_model(
            x_train, y_train, x_test, residual_error, number_irrelevant_variables, n_neighbors=start)
        non_linear_noise.append(np.mean(residual_error))
        non_linear_squared_bias.append(np.mean(squared_bias))
        non_linear_variance.append(np.mean(variance))
        non_linear_error.append(np.mean(error))

    """
    Plot results
    """
    x = range(start, n_samples)
    plt.figure()
    plt.plot(x, linear_error, label="Mean error")
    plt.plot(x, linear_variance, label="Mean variance")
    plt.plot(x, linear_noise, label="Mean residual error")
    plt.plot(x, linear_squared_bias, label="Mean squared Bias")
    plt.xlabel("Learning Sample size")
    plt.ylabel("Mean error")
    plt.legend()
    plt.savefig("Mean_LS_Linear.pdf")

    plt.figure()
    plt.plot(x, non_linear_error, label="Mean error")
    plt.plot(x, non_linear_variance, label="Mean variance")
    plt.plot(x, non_linear_noise, label="Mean residual error")
    plt.plot(x, non_linear_squared_bias, label="Mean squared Bias")
    plt.xlabel("Learning Sample size")
    plt.ylabel("Mean error")
    plt.legend()
    plt.savefig("Mean_LS_Non_Linear.pdf")


def mean_model_complexity(n_samples, n_sets, number_irrelevant_variables, start, end):
    """
    Variables initialisation
    """
    linear_noise = []
    linear_squared_bias = []
    linear_variance = []
    linear_error = []

    non_linear_noise = []
    non_linear_squared_bias = []
    non_linear_variance = []
    non_linear_error = []

    """
    Compute Mean
    """
    for complexity in range(start, end):
        (x_train, y_train, x_test, y_test) = make_data(n_samples, n_sets)
        residual_error = compute_noise(x_train, y_train)

        """
        Linear Regression
        """
        (residual_error, squared_bias, variance, error) = linear_model(
            x_train, y_train, x_test, residual_error, number_irrelevant_variables, alpha=complexity)
        linear_noise.append(np.mean(residual_error))
        linear_squared_bias.append(np.mean(squared_bias))
        linear_variance.append(np.mean(variance))
        linear_error.append(np.mean(error))

        """
        Non Linear Regression
        """
        if complexity == 0:
            continue
        (residual_error, squared_bias, variance, error) = non_linear_model(x_train, y_train,
                                                                           x_test, residual_error, number_irrelevant_variables, n_neighbors=complexity)
        non_linear_noise.append(np.mean(residual_error))
        non_linear_squared_bias.append(np.mean(squared_bias))
        non_linear_variance.append(np.mean(variance))
        non_linear_error.append(np.mean(error))

    """
    Plot results
    """
    x = range(start, end)
    plt.figure()
    plt.plot(x, linear_error, label="Mean error")
    plt.plot(x, linear_variance, label="Mean variance")
    plt.plot(x, linear_noise, label="Mean residual error")
    plt.plot(x, linear_squared_bias, label="Mean squared Bias")
    plt.xlabel("Complexity of the Model")
    plt.ylabel("Mean error")
    plt.legend()
    plt.savefig("Mean_Complexity_linear.pdf")

    # Reverse because Knn is more complex when k is small
    x = range(start + 1, end)
    plt.figure()
    plt.plot(x, non_linear_error[::-1], label="Mean error")
    plt.plot(x, non_linear_variance[::-1], label="Mean variance")
    plt.plot(x, non_linear_noise[::-1], label="Mean residual error")
    plt.plot(x, non_linear_squared_bias[::-1], label="Mean squared Bias")
    plt.xlabel("Complexity of the Model")
    plt.ylabel("Mean error")
    plt.legend()
    plt.savefig("Mean_Complexity_non_linear.pdf")


def mean_irrelevant_variables(n_samples, n_sets, number_irrelevant_variables):
    """
    Variable initialisation
    """
    linear_noise = []
    linear_squared_bias = []
    linear_variance = []
    linear_error = []

    non_linear_noise = []
    non_linear_squared_bias = []
    non_linear_variance = []
    non_linear_error = []

    """
    Compute mean
    """
    for irr_var in range(number_irrelevant_variables):
        (x_train, y_train, x_test, y_test) = make_data(n_samples, n_sets)
        residual_error = compute_noise(x_train, y_train)

        """
        Linear Regression
        """
        (residual_error, squared_bias, variance, error) = linear_model(
            x_train, y_train, x_test, residual_error, irr_var)
        linear_noise.append(np.mean(residual_error))
        linear_squared_bias.append(np.mean(squared_bias))
        linear_variance.append(np.mean(variance))
        linear_error.append(np.mean(error))

        """
        Non Linear Regression
        """
        (residual_error, squared_bias, variance, error) = non_linear_model(
            x_train, y_train, x_test, residual_error, irr_var)
        non_linear_noise.append(np.mean(residual_error))
        non_linear_squared_bias.append(np.mean(squared_bias))
        non_linear_variance.append(np.mean(variance))
        non_linear_error.append(np.mean(error))

    """
    Plot results
    """
    x = range(number_irrelevant_variables)
    plt.figure()
    plt.plot(x, linear_error, label="Mean error")
    plt.plot(x, linear_variance, label="Mean variance")
    plt.plot(x, linear_noise, label="Mean residual error")
    plt.plot(x, linear_squared_bias, label="Mean squared Bias")
    plt.xlabel("Number of irrelevant variables")
    plt.ylabel("Mean error")
    plt.legend()
    plt.savefig("Mean_irr_var_Linear.pdf")

    plt.figure()
    plt.plot(x, non_linear_error, label="Mean error")
    plt.plot(x, non_linear_variance, label="Mean variance")
    plt.plot(x, non_linear_noise, label="Mean residual error")
    plt.plot(x, non_linear_squared_bias, label="Mean squared Bias")
    plt.xlabel("Number of irrelevant variables")
    plt.ylabel("Mean error")
    plt.legend()
    plt.savefig("Mean_irr_var_Non_Linear.pdf")


if __name__ == "__main__":
    n_samples = 2000
    n_sets = 50
    number_irrelevant_variables = 0

    """
    test error
    """
    compute_error(n_samples, n_sets, number_irrelevant_variables, alpha=8)

    """
    LS size
    """
    # decrease n_samples for simplicity in the plots
    n_samples = 100
    start = 1
    mean_size_LS(n_samples, n_sets, 0, start)

    """
    model complexity
    """
    # n_samples=1250 for simplicity and speed
    n_samples = 1250
    start_complexity = 0
    end_complexity = 100
    mean_model_complexity(n_samples, n_sets, 0,
                          start_complexity, end_complexity)

    """
    number of irrelevant variables
    """
    # decrease n_samples for simplicity and speed
    n_samples = 100
    number_irrelevant_variables = 75
    mean_irrelevant_variables(n_samples, n_sets, number_irrelevant_variables)
