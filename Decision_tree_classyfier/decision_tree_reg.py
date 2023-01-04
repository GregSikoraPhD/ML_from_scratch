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
                 MSE_gain: Optional[float] = None, left=None, right=None, side: Optional[str] = None,
                 *, value: Optional[int] = None) -> None:
        self.depth = depth
        self.n_samples = n_samples
        self.threshold = threshold
        self.MSE_gain = MSE_gain
        self.left = left
        self.right = right
        self.side = side
        self.value = value

    def is_leaf_node(self) -> bool:
        return self.value is not None


class DecisionTreeReg:
    """class for a decision tree classifier object."""
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
        acctual_MSE = mse(y)

        # stopping criteria
        if (
                depth >= self.max_depth
                or acctual_MSE <= self.min_MSE
                or n_samples < self.min_samples_split
        ):
            leaf_value = np.mean(y)
            leaf_node = Node(depth=depth, side=side, value=leaf_value)
            self.structure.append(leaf_node)
            return leaf_node

        # greedy search
        best_thresh, MSE_gain = self._best_criteria(x, y, acctual_MSE)

        # grow the children from split
        left_idxs, right_idxs = self._split(x, best_thresh)
        # side = 'left'
        left = self._grow_tree(x[left_idxs], y[left_idxs], depth + 1, side='left')
        # side = 'right'
        right = self._grow_tree(x[right_idxs], y[right_idxs], depth + 1, side='right')
        if depth == 0:
            node = Node(depth, n_samples, best_thresh, MSE_gain, left, right, 'root')
            self.structure.append(node)
            return node
        else:
            node = Node(depth, n_samples, best_thresh, MSE_gain, left, right, side)
            self.structure.append(node)
            return node

    def _best_criteria(self, x: np.ndarray, y: np.ndarray, acctual_MSE: float) -> Tuple:
        split_thresh = None
        thresholds = np.unique(x)
        for threshold in thresholds:
            gain_MSE = self._MSE_gain(y, x, threshold)
            if gain_MSE < acctual_MSE:
                best_MSE = gain_MSE
                split_thresh = threshold

        return split_thresh, best_MSE

    # TODO
    def _MSE_gain(self, y: np.ndarray, X_column: np.ndarray, split_thresh: float) -> float:
        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) <= 1 or len(right_idxs) <= 1:
            return 0

        # max of child MSE errors
        mse_l, mse_r = mse(y[left_idxs]), mse(y[right_idxs])
        max_MSE = max(mse_l, mse_r)
        return max_MSE

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

    # TODO
    def export_text(self, feature_names: List[str]) -> None:
        if self.structure:
            for node in self.structure[::-1]:
                preff = '|   ' * (node.depth - 1) + '|--- '
                if node.n_samples:
                    if node.depth == 0:
                        print(f'''Root node: n_samples = {node.n_samples}, 
            splitting_feature = {feature_names[node.feature]}, 
            splitting_threshold = {node.threshold}''')
                    else:
                        print(preff + f'''Node {node.side}: n_samples = {node.n_samples}, 
            splitting_feature = {feature_names[node.feature]}, 
            splitting_threshold = {node.threshold}''')
                else:
                    print(preff + f'Leaf node {node.side}: {node.value}')



