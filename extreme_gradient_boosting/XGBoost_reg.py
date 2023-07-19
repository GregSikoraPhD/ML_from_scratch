import numpy as np
from sklearn.base import BaseEstimator
from typing import Tuple


class XGBoost:
    """
    XGBoost: Extreme Gradient Boosting.

    Parameters:
        n_estimators (int): Number of trees (boosting iterations).
        max_depth (int): Maximum depth of each decision tree.
        learning_rate (float): Learning rate or shrinkage factor.
        weak_learner (BaseEstimator): Weak learner used as base estimator.
        loss (str or callable): Loss function to optimize during boosting. Can be a string representing
                                a built-in loss function ('ls', 'lad', 'huber') or a custom loss function
                                that takes (y_true, y_pred) as input and returns the gradient and hessian.

    Attributes:
        n_estimators (int): Number of trees (boosting iterations).
        max_depth (int): Maximum depth of each decision tree.
        learning_rate (float): Learning rate or shrinkage factor.
        weak_learner (BaseEstimator): Weak learner used as base estimator.
        loss (str or callable): Loss function to optimize during boosting. Can be a string representing
                                a built-in loss function ('ls', 'lad', 'huber') or a custom loss function
                                that takes (y_true, y_pred) as input and returns the gradient and hessian.
        estimators (list): List of trained decision trees.

    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 3,
                 learning_rate: float = 0.1, weak_learner: BaseEstimator = None,
                 loss='ls'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.weak_learner = weak_learner
        self.loss = loss
        self.estimators = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the XGBoost model to the given training data.

        Parameters:
            X (np.ndarray): Training data features.
            y (np.ndarray): Training data labels.

        Returns:
            None
        """
        y_pred = np.zeros_like(y)
        for _ in range(self.n_estimators):
            gradient, hessian = self._calculate_gradient_hessian(y, y_pred)
            estimator = self.weak_learner(max_depth=self.max_depth)
            estimator.fit(X, -gradient / hessian)  # Fit with negative gradient divided by Hessian
            self.estimators.append(estimator)
            y_pred += self.learning_rate * estimator.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for the given input data.

        Parameters:
            X (np.ndarray): Input data features.

        Returns:
            np.ndarray: Predicted values.
        """
        y_pred = np.zeros(len(X))
        for estimator in self.estimators:
            y_pred += self.learning_rate * estimator.predict(X)
        return y_pred

    def _calculate_gradient_hessian(self, y: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the gradient and Hessian of the loss function.

        Parameters:
            y (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            tuple[np.ndarray, np.ndarray]: Gradient and Hessian values.
        """
        if isinstance(self.loss, str):
            if self.loss == 'ls':  # Least Squares Loss
                gradient = y_pred - y
                hessian = np.ones_like(y)
            elif self.loss == 'lad':  # Least Absolute Deviation Loss
                gradient = np.sign(y_pred - y)
                hessian = np.ones_like(y)
            elif self.loss == 'huber':  # Huber Loss
                delta = 1.0
                residual = y_pred - y
                gradient = np.where(np.abs(residual) <= delta, residual, delta * np.sign(residual))
                hessian = np.where(np.abs(residual) <= delta, 1.0, delta / np.abs(residual))
            else:
                raise ValueError(f"Invalid loss function: {self.loss}")
        elif callable(self.loss):
            gradient, hessian = self.loss(y, y_pred)
            if not isinstance(gradient, np.ndarray) or not isinstance(hessian, np.ndarray):
                raise ValueError("Custom loss function must return gradient and hessian as numpy arrays.")
            if gradient.shape != y.shape or hessian.shape != y.shape:
                raise ValueError("Shape mismatch: gradient and hessian should have the same shape as y.")
        else:
            raise ValueError("Invalid loss function type. Please provide a valid loss function.")

        return gradient, hessian
