import numpy as np


class LinearRegression:
    """
    This class represents a simple linear regression model which uses gradient descent for training.

    Attributes:
        lr: A float representing the learning rate for gradient descent.
        n_iters: An integer for the number of iterations for the gradient descent loop.
        weights: A numpy array of feature weights (will be initialized in the fit method).
        bias: A float for the bias term (will be initialized in the fit method).
    """

    def __init__(self, lr: float = 0.001, n_iters: int = 1000):
        """
        The constructor for LinearRegression class.

        Parameters:
            lr: The learning rate for gradient descent (default is 0.001).
            n_iters: The number of iterations for the gradient descent loop (default is 1000).
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights: np.ndarray = None
        self.bias: float = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains the linear regression model on the given data.

        Parameters:
            X: A 2D numpy array where each row is a data sample and each column is a feature.
            y: A 1D numpy array with target values for each sample in X.
        """
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descend
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # derivatives for gradient
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            # update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method makes predictions using the trained linear regression model.

        Parameters:
            X: A 2D numpy array where each row is a data sample and each column is a feature.

        Returns:
            A 1D numpy array with the predicted target value for each sample in X.
        """
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

