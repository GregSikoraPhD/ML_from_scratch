import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """function for model accuracy calculation."""
    return np.sum(y_true == y_pred) / len(y_true)


class DecisionStump:
    """class for a decision stump classifier object."""
    def __init__(self) -> None:
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]

        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions


class AdaboostCls:
    """class for a AdaBoost classifier object."""
    def __init__(self, n_cls: int = 5) -> None:
        self.n_cls = n_cls
        self.clss = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape

        # init weights
        w = np.full(n_samples, (1/n_samples))

        for _ in range(self.n_cls):
            cls = DecisionStump()

            min_error = float('inf')
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    misclassified = w[y != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        min_error = error
                        cls.polarity = p
                        cls.threshold = threshold
                        cls.feature_idx = feature_i

            EPS = 1e-10
            cls.alpha = 0.5 * np.log((1-error) / (error+EPS))

            predictions = cls.predict(X)

            w *= np.exp(-cls.alpha * y * predictions)
            w /= np.sum(w)

            self.clss.append(cls)

    def predict(self, X: np.ndarray) -> np.ndarray:
        cls_preds = [cls.alpha * cls.predict(X) for cls in self.clss]
        y_pred = np.sum(cls_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred
