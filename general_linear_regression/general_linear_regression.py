import numpy as np
from linear_regression import LinearRegression


class GeneralLinearRegression:
    """
    This class represents a general (= multivariate, i.e. multiple input features and multiple output values)
    linear regression model which uses gradient descent for training.

    Attributes:
        lr: A float representing the learning rate for gradient descent.
        n_iters: An integer for the number of iterations for the gradient descent loop.
        weights: A numpy array of feature weights (will be initialized in the fit method).
        bias: A numpy array for the bias terms (will be initialized in the fit method).
        regressors: A list of regressors for each output coordinate
        n_coordinates: Number of output coordinates
    """

    def __init__(self, lr: float = 0.001, n_iters: int = 1000):
        """
        The constructor for GeneralLinearRegression class.

        Parameters:
            lr: The learning rate for gradient descent (default is 0.001).
            n_iters: The number of iterations for the gradient descent loop (default is 1000).
            regressors: A list of regressors for each output coordinate
            n_coordinates: Number of output coordinates
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights: np.ndarray = None
        self.bias: np.ndarray = None
        self.regressors = []
        self.n_coordinates: None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        This method trains the general linear regression model on the given data by iterating through each coordinates
        of the output variable and fitting a multiple linear regression

        Parameters:
            X: A 2D numpy array where each row is a data sample and each column is a feature.
            Y: A 2D numpy array where each row is a vector output for a sample and each column is a output coordinate.
        """
        # init parameters
        n_samples, n_features = X.shape
        self.n_coordinates = Y.shape[1]

        self.weights = np.zeros(self.n_coordinates, n_features)
        self.bias = np.zeros(self.n_coordinates)

        # iterating through output coordinates and fitting a multiple linear regression for a single coordinate
        for i in range(self.n_coordinates):
            regressor = LinearRegression(lr=self.lr, n_iters=self.n_iters)
            regressor.fit(X, Y[:, i])
            self.weights[i, :] = regressor.weights
            self.bias[i] = regressor.bias
            self.regressors.append(regressor)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method makes predictions using the trained general linear regression model.

        Parameters:
           X: A 2D numpy array where each row is a data sample and each column is a feature.

        Returns:
           A 2D numpy array each row is a vector output for a sample and each column is a output coordinate.
        """
        y_predicted = []
        # iterating through output coordinates and predicting single output for all samples
        for i in range(self.n_coordinates):
            y_i_predicted = self.regressors[i].predict(X)
            y_predicted.append(y_i_predicted)
        return y_predicted






