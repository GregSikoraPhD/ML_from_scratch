import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from decision_tree_classifier.decision_tree_cls import DecisionTreeCls
from typing import Optional, Tuple
from statsmodels.distributions.empirical_distribution import ECDF


def bootstrap_sample(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """function for bootstrap subsampling."""
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxs], y[idxs]


def most_common_label(y: np.ndarray) -> int:
    """function for most common label choice."""
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """function for model accuracy calculation."""
    return np.sum(y_true == y_pred) / len(y_true)


class RandomForrestCls:
    """class for a random forrest classifier object."""
    def __init__(self, n_trees: int = 100, min_samples_split: int = 2, max_depth: int = 100,
                 n_feats: Optional[int] = None) -> None:
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeCls(min_samples_split=self.min_samples_split,
                                   max_depth=self.max_depth, n_feats=self.n_feats)
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict_value(self, X: np.ndarray) -> np.ndarray:
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)

    def predict_distribution(self, X: np.ndarray, plot: Optional[bool] = False) -> np.ndarray:
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_ECDF_preds = [ECDF(tree_pred) for tree_pred in tree_preds]
        if plot:
            for x, pred, ecdf in zip(X, tree_preds, y_ECDF_preds):
                self._plot_ECDF(pred, ecdf, str(x))
        return np.array(y_ECDF_preds)

    @staticmethod
    def _plot_ECDF(x: np.ndarray, ecdf_x: ECDF, x_label: str, n_args: int = 100) -> None:
        args_ECDF = np.linspace(min(x), max(x), n_args)
        plt.plot(args_ECDF, ecdf_x(args_ECDF))
        plt.xlabel(x_label)
        plt.ylabel('ECDF of predictions for ' + x_label)
        plt.ylim([0, 1])
        plt.show()
