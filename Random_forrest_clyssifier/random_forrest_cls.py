import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from decision_tree_cls import DecisionTreeCls
from typing import List, Optional, Tuple
from statsmodels.distributions.empirical_distribution import ECDF


def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxs], y[idxs]


def most_common_label(y: np.ndarray) -> int:
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


class RandomForrestCls:

    def __init__(self, n_trees: int = 100, min_samples_split: int = 2, max_depth: int = 100,
                 n_feats: Optional[int] = None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeCls(min_samples_split=self.min_samples_split,
                                   max_depth=self.max_depth, n_feats=self.n_feats)
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict_value(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)

    def predict_distribution(self, X, plot=False):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_ECDF_preds = [ECDF(tree_pred) for tree_pred in tree_preds]
        if plot:
            for x, pred, ecdf in zip(X, tree_preds, y_ECDF_preds):
                self._plot_ECDF(pred, ecdf, str(x))
        return np.array(y_ECDF_pred)

    @staticmethod
    def _plot_ECDF(x: np.ndarray, ecdf_x: np.ndarray, x_label: str, n_args: int=100):
        args_ECDF = np.linspace(min(x), max(x), n_args)
        plt.plot(args_ECDF, ecdf_x)
        plt.xlabel(x_label)
        plt.ylabel('ECDF of predictions for ' + x_label)
        plt.show()
