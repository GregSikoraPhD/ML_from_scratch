import numpy as np
from sympy import symbols, diff
from sympy.core.symbol import Symbol
from sympy.core.expr import Expr
from typing import Callable


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """function for model accuracy calculation."""
    return np.sum(y_true == y_pred) / len(y_true)


class GradientBoostingRegressor:
    """
    Gradient Boosting Regressor using weak learners.

    Parameters:
        n_estimators (int): The number of boosting stages to perform (default: 100).
        learning_rate (float): The learning rate of the boosting process (default: 0.1).
        base_model: The base regression model to use as a weak learner (default: None).
        loss_func: The callable formula for the loss function (default: 'mse').
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, base_model=None,
                 loss_func: Callable[[Symbol, Symbol], Expr] = lambda y, y_pred: 0.5*(y - y_pred)**2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_model = base_model
        self.loss_func = loss_func
        self.estimators = []

    def _custom_loss_gradient(self, loss_func: Callable[[Symbol, Symbol], Expr], y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the numerically gradient for a custom symbolic loss function.

        Parameters:
            y (ndarray): The target values. Shape (n_samples,).
            y_pred (ndarray): The predicted values. Shape (n_samples,).

        Returns:
            ndarray: The gradient of the custom loss function. Shape (n_samples,).
        """
        # Compute the gradient of the loss function with respect to y_pred
        gradient_func_symbolic = self._symbolic_gradient(loss_func)

        # Compute the gradient values in y_pred
        gradient_func_values = [gradient_func_symbolic.subs([('y', i), ('y_pred', j)]) for i, j in zip(y, y_pred)]

        return np.array(gradient_func_values)

    @staticmethod
    def _symbolic_gradient(loss_func: Callable[[Symbol, Symbol], Expr]) -> Expr:
        """
          Calculate the symbolic gradient for a custom symbolic loss function.

          Parameters:
              loss_func: Callable[[symbols.Symbol], symbols.Expr]: Callable custom function.

          Returns:
              Expr:: The symbolic gradient of the custom loss function.
          """
        x, y = symbols('y_pred y')
        return diff(loss_func(y, x), x)

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
            negative_gradient = - self._custom_loss_gradient(self.loss_func, y, predictions)
            model = self.base_model() if self.base_model else None
            model.fit(X, negative_gradient)
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

