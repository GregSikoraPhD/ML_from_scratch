import numpy as np


class GradientBoostingRegressor:
    """
    Gradient Boosting Regressor using weak learners.

    Parameters:
        n_estimators (int): The number of boosting stages to perform (default: 100).
        learning_rate (float): The learning rate of the boosting process (default: 0.1).
        base_model: The base regression model to use as a weak learner (default: None).
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, base_model=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_model = base_model
        self.estimators = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the gradient boosting regressor to the training data.

        Parameters:
            X (ndarray): The training input samples. Shape (n_samples, n_features).
            y (ndarray): The target values. Shape (n_samples,).
        """
        self.estimators = []
        predictions = np.zeros_like(y, dtype=np.float64)
        for _ in range(self.n_estimators):
            residual = y - predictions
            model = self.base_model() if self.base_model else None
            model.fit(X, residual)
            self.estimators.append(model)
            tree_predictions = model.predict(X)
            predictions += self.learning_rate * tree_predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for the input samples.

        Parameters:
            X (ndarray): The input samples. Shape (n_samples, n_features).

        Returns:
            ndarray: The predicted target values. Shape (n_samples,).
        """
        predictions = np.zeros(X.shape[0], dtype=np.float64)
        for estimator in self.estimators:
            predictions += self.learning_rate * estimator.predict(X)
        return predictions



