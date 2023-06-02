from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
from pandas import DataFrame


def rotation_base(theta: float) -> Tuple[np.ndarray, np.ndarray]:
    """function for a new base after rotation."""
    theta = np.radians(theta)
    x1, y1 = np.cos(theta), np.sin(theta)
    x2, y2 = -y1, x1
    return np.array([x1, y1]), np.array([x2, y2])


def change_coordinates(theta: float, data: np.ndarray) -> np.ndarray:
    """function for a change of data in a new coordinate system after rotation."""
    b1, b2 = rotation_base(theta)
    data_new = np.linalg.inv(np.column_stack((b1, b2))).dot(data.T)
    return data_new.T


def entropy(y: np.ndarray) -> float:
    """function for entropy calculation."""
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """function for model accuracy calculation."""
    return np.sum(y_true == y_pred) / len(y_true)


class Node:
    """class for a Node object"""
    def __init__(self, depth: Optional[int] = None, n_samples: Optional[int] = None, rotation_angle: Optional[int] = 0,
                 feature: Optional[int] = None, threshold: Optional[float] = None,
                 information_gain: Optional[float] = None, left=None, right=None, side: Optional[str] = None, *,
                 value: Optional[int] = None) -> None:
        self.depth = depth
        self.n_samples = n_samples
        self.rotation_angle = rotation_angle
        self.feature = feature
        self.threshold = threshold
        self.information_gain = information_gain
        self.left = left
        self.right = right
        self.side = side
        self.value = value

    def is_leaf_node(self) -> bool:
        return self.value is not None


class RotationTreeCls:
    """class for a decision tree classifier object."""
    def __init__(self, min_samples_split: int = 2, max_depth: int = 100, n_feats: Optional[int] = None) -> None:
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
        self.structure = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # grow tree
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0, side: str = 'root') -> Node:
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (
                depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            leaf_node = Node(depth=depth, side=side, value=leaf_value)
            self.structure.append(leaf_node)
            return leaf_node

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedy search
        best_angle, best_feat, best_thresh, information_gain = self._best_criteria(X, y, feat_idxs)
        X = change_coordinates(best_angle, X)

        # grow the children from split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        # side = 'left'
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1, side='left')
        # side = 'right'
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1, side='right')
        if depth == 0:
            node = Node(depth, n_samples, best_angle, best_feat, best_thresh, information_gain, left, right, 'root')
            self.structure.append(node)
            return node
        else:
            node = Node(depth, n_samples, best_angle, best_feat, best_thresh, information_gain, left, right, side)
            self.structure.append(node)
            return node

    def _best_criteria(self, X: np.ndarray, y: np.ndarray, feat_idxs: np.ndarray) -> Tuple:
        best_gain = -1
        split_idx, split_thresh = None, None
        for theta in range(0, 180, 5):
            X = change_coordinates(theta, X)
            for feat_idx in feat_idxs:
                X_column = X[:, feat_idx]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    gain = self._information_gain(y, X_column, threshold)

                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feat_idx
                        split_thresh = threshold
                        rotation_angle = theta

        return rotation_angle, split_idx, split_thresh, best_gain

    def _information_gain(self, y: np.ndarray, X_column: np.ndarray, split_thresh: float) -> float:
        # parent entropy
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # weighted avg child entropy
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # return information gain
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column: np.ndarray, split_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def predict(self, X: np.ndarray) -> np.ndarray:
        # predict
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: np.ndarray, node: Node) -> int:
        if node.is_leaf_node():
            return node.value
        x = change_coordinates(node.rotation_angle, x)
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    @staticmethod
    def _most_common_label(y: np.ndarray) -> int:
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def export_text(self, feature_names: List[str]) -> None:
        if self.structure:
            for node in self.structure[::-1]:
                preff = '|   ' * (node.depth - 1) + '|--- '
                if node.n_samples:
                    if node.depth == 0:
                        print(f'''Root node: n_samples = {node.n_samples},
            rotation_angle = {node.rotation_angle},
            splitting_feature = {feature_names[node.feature]}, 
            splitting_threshold = {node.threshold}''')
                    else:
                        print(preff + f'''Node {node.side}: n_samples = {node.n_samples}, 
            splitting_feature = {feature_names[node.feature]}, 
            rotation_angle = {node.rotation_angle},
            splitting_threshold = {node.threshold}''')
                else:
                    print(preff + f'Leaf node {node.side}: {node.value}')

    def feature_importance(self, feature_names: List[str]) -> DataFrame:
        if self.structure:
            feature = []
            importance = []
            for i, node in enumerate(self.structure[::-1]):
                if not node.is_leaf_node():
                    feature.append(feature_names[node.feature])
                    importance.append(node.information_gain)
            feature_importance = DataFrame([feature, importance]).T
            feature_importance.columns = {'feature_importance', 'feature_name'}
            return feature_importance



