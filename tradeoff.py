import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.utils import check_random_state


def f(X):
    """
    Generates the output function taking the X input sets of a sample

    parameters:
    - X : numpy.ndarray
        The set of inputs in the sample

    Returns:
    - A numpy.ndarray containing the outputs of the sample
    """
    return np.sin(X) * np.exp((-X**2)/16)


def make_data(n_samples, n_sets, random_state=None):
    """
    Generates a set of N samples (xr, y)

    parameters:
    - n_samples : int
        the number N of samples to generate
    - n_sets : int
        the number of learning sets
    - random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns:
    - x_train: numpy.ndarray
        contains the random features value uniformely distributed
    - y train: numpy.ndarray
        contains the output for each observations taking into account a given function f(x)
    - x_test: numpy.ndarray
        contains the features value of the test set
    - y_test: numpy.ndarray
        contains the output values of the test set
    """
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
    """
    Given a trained linear model and a test set, returns its total error added to its decomposition in noise, variance and squared bias

    parameters:
    - x_train : numpy.ndarray
        contains the random features value uniformely distributed
    - y train : numpy.ndarray
        contains the output for each observations taking into account a given function f(x)
    - x_test : numpy.ndarray
        contains the features value of the test set
    - residual_error : float
        the noise computed on x_train and y_train
    - number_irrelevant_variables : int
        a number of irrelevant variables to add to the model
    - plot : bool, optinal
        True if we want to generate plot our results, false otherwise
    - alpha : int
        the alpha parameters of the Lasso algorithm
    Returns:
    - noise : numpy.ndarray
        A new numpy.ndarray containing the noise
    - squared_bias : numpy.ndarray
        A new numpy.ndarray containing the squared bias
    - variance : numpy.ndarray
        A new numpy.ndarray containing the variance
    - error : numpy.ndarray
        the total error of the model, computed as the sum of noise, squared bias and variance
    """
    if alpha == 0:
        # alpha = 0 is the same as a LinearRegression
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
    """
    Given a trained KNN model and a test set, returns its total error added to its decomposition in noise, variance and squared bias

    parameters:
    - x_train : numpy.ndarray
        contains the random features value uniformely distributed
    - y train : numpy.ndarray
        contains the output for each observations taking into account a given function f(x)
    - x_test : numpy.ndarray
        contains the features value of the test set
    - residual_error : float
        the noise computed on x_train and y_train
    - number_irrelevant_variables : int
        a number of irrelevant variables to add to the model
    - plot : bool, optinal
        True if we want to generate plot our results, false otherwise
    - n_neighbors : int
        the chosen k number of neighbors to generate our model

    Returns:
    - noise : numpy.ndarray
        A new numpy.ndarray containing the noise
    - squared_bias : numpy.ndarray
        A new numpy.ndarray containing the squared bias
    - variance : numpy.ndarray
        A new numpy.ndarray containing the variance
    - error : numpy.ndarray
        the total error of the model, computed as the sum of noise, squared bias and variance
    """
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
    """
    Computes the variance of the elements a given array_like

    parameters:
    - predictions : List
        the values on which we want the variance

    Returns:
    - A new numpy.ndarray containing the variance
    """
    return np.var(predictions, 0)


def compute_squared_bias(x, prediction):
    """
    Computes the squared bias of the elements a given array_like, the error between the Bayes model and the average one at a given point x.

    parameters:
    - predictions : List
        the values on which we want the variance
    - x : numpy.ndarray
        the values of the inputs of the Bayes model
    Returns:
    - A new numpy.ndarray containing the squared bias
    """
    bayes_model = f(x)
    return (bayes_model - np.mean(prediction, 0))**2


def compute_noise(x, y):
    """
    Computes the noise, a quantification of variation between a array y and its corresponding bayes model computed on the x inputs.

    parameters:
    - x : numpy.ndarray
        The values of the inputs of the Bayes model
    - y : numpy.ndarray
        the corresponding outputs of the sample

    Returns:
    A new numpy.ndarray containing the noise
    """
    residual_error = []
    for i in range(x.shape[0]):
        bayes_model = f(x[i])
        residual_error.append((y[i] - bayes_model)**2)
    return np.mean(np.array(residual_error))


def add_irrelevant_variables(X, number_irrelevant_variables, random_state=None):
    """
    Add a number "number_irrelevant_variables"  of irrelevant variables to the problem defined by the initial input values X.

    parameters:
    - X : numpy.ndarray
        The set of inputs in the sample
    - number_irrelevant_variables : int
        a number of irrelevant variables to add to the model
    - random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns:
    - A ndarray with the initial inputs X in which we added a set of irrelevant inputs.

    """
    if number_irrelevant_variables == 0:
        return X.reshape(-1, 1)
    random = check_random_state(random_state)
    variables = np.zeros((X.shape[0], 1 + number_irrelevant_variables))
    for i in range(X.shape[0]):
        variables[i] = np.concatenate(
            ([X[i]], random.uniform(-10, 10, number_irrelevant_variables)))
    return variables.reshape(-1, 1 + number_irrelevant_variables)


def compute_error(n_samples, n_sets, number_irrelevant_variables, alpha=0, n_neighbors=5):
    """
    Given a number of sample and a number of learning sets, generates the N samples and compute the error on a linear and a non-linear regression

    parameters:
    - n_samples : int
        the number N of samples to generate
    - n_sets : int
        the number of learning sets
    - number_irrelevant_variables : int
        a number of irrelevant variables to add to the model
    - alpha : int
        The alpha parameters of the lasso algorithm
    - n_neighbors : int
        the chosen k number of neighbors to generate our model

    Returns:
    /
    """
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
    Computes and plost the mean values of the noise, squared bias and variance in function of the size of the learning samples on a linear and a non-linear model.

    parameters:
    - n_samples : int
        the number N of samples to generate
    - n_sets : int
        the number of learning sets
    - number_irrelevant_variables : int
        a number of irrelevant variables to add to the model
    - start : int
        number of neighbors minimum

    Returns:
    /
    """

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

        if size < 10:
            continue
        """
        Non Linear Regression
        """
        (residual_error, squared_bias, variance, error) = non_linear_model(
            x_train, y_train, x_test, residual_error, number_irrelevant_variables, n_neighbors=10)
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

    x = range(start + 9, n_samples)
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
    Computes and plots the mean values of the noise, squared bias and variance in function of the model complexity on a linear and a non-linear model.

    parameters:
    - n_samples : int
        the number N of samples to generate
    - n_sets : int
        the number of learning sets
    - number_irrelevant_variables : int
        a number of irrelevant variables to add to the model
    - start : int
        the start number of neighbors or the alpha parameters of the regression models
    - end: int
        the end number of neighbors or the alpha parameters of the regression models

    Returns:
    /
    """

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
    plt.plot(x, non_linear_error, label="Mean error")
    plt.plot(x, non_linear_variance, label="Mean variance")
    plt.plot(x, non_linear_noise, label="Mean residual error")
    plt.plot(x, non_linear_squared_bias, label="Mean squared Bias")
    plt.xlabel("Complexity of the Model")
    plt.ylabel("Mean error")
    plt.legend()
    plt.savefig("Mean_Complexity_non_linear.pdf")


def mean_irrelevant_variables(n_samples, n_sets, number_irrelevant_variables):
    """
    Computes and plots the mean values of the noise, squared bias and variance in function of the number of irrelevant variables on a linear and a non-linear model.

    parameters:
    - n_samples : int
        the number N of samples to generate
    - n_sets : int
        the number of learning sets
    - number_irrelevant_variables : int
        a number of irrelevant variables to add to the model

    Returns:
    /
    """

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
            x_train, y_train, x_test, residual_error, irr_var, alpha=3)
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
    n_samples = 150
    start = 1
    mean_size_LS(n_samples, n_sets, 0, start)

    """
    model complexity
    """
    # n_samples=1250 for simplicity and speed
    n_samples = 1250
    start_complexity = 0
    end_complexity = 150
    mean_model_complexity(n_samples, n_sets, 0,
                          start_complexity, end_complexity)

    """
    number of irrelevant variables
    """
    # decrease n_samples for simplicity and speed
    n_samples = 150
    number_irrelevant_variables = 75
    mean_irrelevant_variables(n_samples, n_sets, number_irrelevant_variables)
