from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
from pandas import DataFrame


def mse(y: np.ndarray) -> float:
    """function for MSE calculation."""
    avg = np.mean(y)
    squared_errors = (y - avg)**2
    return np.mean(squared_errors)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """function for model accuracy calculation."""
    return np.sum(y_true == y_pred) / len(y_true)


class Node:
    """class for a Node object"""
    def __init__(self, depth: Optional[int] = None, n_samples: Optional[int] = None, threshold: Optional[float] = None,
                 actual_MSE: Optional[float] = None, childs_MSE: Optional[float] = None, left=None, right=None,
                 side: Optional[str] = None, *, value: Optional[int] = None) -> None:
        self.depth = depth
        self.n_samples = n_samples
        self.threshold = threshold
        self.actual_MSE = actual_MSE
        self.childs_MSE = childs_MSE
        self.left = left
        self.right = right
        self.side = side
        self.value = value

    def is_leaf_node(self) -> bool:
        return self.value is not None


class DecisionTreeReg:
    """class for a (only univariate) decision tree regressor object."""
    def __init__(self, min_samples_split: int = 2, min_MSE: float = 1.0, max_depth: int = 100) -> None:
        self.min_samples_split = min_samples_split
        self.min_MSE = min_MSE
        self.max_depth = max_depth
        self.root = None
        self.structure = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        # grow tree
        self.root = self._grow_tree(x, y)

    def _grow_tree(self, x: np.ndarray, y: np.ndarray, depth: int = 0, side: str = 'root') -> Node:
        n_samples = len(x)
        actual_MSE = mse(y)

        # stopping criteria
        if (
                depth >= self.max_depth
                or actual_MSE <= self.min_MSE
                or n_samples < self.min_samples_split
        ):
            leaf_value = np.mean(y)
            leaf_node = Node(depth=depth, n_samples=n_samples, actual_MSE=actual_MSE, side=side, value=leaf_value)
            self.structure.append(leaf_node)
            return leaf_node

        # greedy search
        best_thresh, childs_MSE = self._best_criteria(x, y, actual_MSE)

        # grow the children from split
        left_idxs, right_idxs = self._split(x, best_thresh)
        # side = 'left'
        left = self._grow_tree(x[left_idxs], y[left_idxs], depth + 1, side='left')
        # side = 'right'
        right = self._grow_tree(x[right_idxs], y[right_idxs], depth + 1, side='right')
        if depth == 0:
            node = Node(depth, n_samples, best_thresh, actual_MSE, childs_MSE, left, right, 'root')
            self.structure.append(node)
            return node
        else:
            node = Node(depth, n_samples, best_thresh, actual_MSE, childs_MSE, left, right, side)
            self.structure.append(node)
            return node

    def _best_criteria(self, x: np.ndarray, y: np.ndarray, actual_MSE: float) -> Tuple:
        split_thresh = None
        thresholds = np.unique(x)
        childs_MSE_vec = []
        for threshold in thresholds[:-1]:
            childs_MSE = self._childs_MSE(x, y, threshold)
            childs_MSE_vec.append(childs_MSE)
        best_MSE = min(childs_MSE_vec)
        split_thresh = thresholds[np.argmin(childs_MSE_vec)]

        return split_thresh, best_MSE

    def _childs_MSE(self, x: np.ndarray, y: np.ndarray, split_thresh: float) -> float:
        # generate split
        left_idxs, right_idxs = self._split(x, split_thresh)

        # child MSE errors
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        mse_l, mse_r = mse(y[left_idxs]), mse(y[right_idxs])
        childs_MSE = (n_l/n) * mse_l + (n_r/n) * mse_r

        return childs_MSE

    def _split(self, x: np.ndarray, split_thresh: float) -> Tuple[np.ndarray, ...]:
        left_idxs = np.argwhere(x <= split_thresh).flatten()
        right_idxs = np.argwhere(x > split_thresh).flatten()
        return left_idxs, right_idxs

    def predict(self, X: np.ndarray) -> np.ndarray:
        # predict
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: float, node: Node) -> int:
        if node.is_leaf_node():
            return node.value

        if x <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def export_text(self) -> None:
        if self.structure:
            for node in self.structure[::-1]:
                preff = '|   ' * (node.depth - 1) + '|--- '
                if not node.is_leaf_node():
                    if node.depth == 0:
                        print(f'''Root node: n_samples = {node.n_samples},
            actual_MSE = {node.actual_MSE},
            childs_MSE = {node.childs_MSE},
            splitting_threshold = {node.threshold}''')
                    else:
                        print(preff + f'''Node {node.side}: n_samples = {node.n_samples},
            actual_MSE = {node.actual_MSE},
            childs_MSE = {node.childs_MSE},
            splitting_threshold = {node.threshold}''')
                else:
                    print(preff + f'Leaf node {node.side}: {node.value}')