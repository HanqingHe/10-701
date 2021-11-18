from typing import Tuple

import numpy as np
import pandas as pd


class DecisionStump:
    def __init__(self, epsilon: float = 1e-6):
        r"""A depth-1 decision tree classifier

        Args:
            epsilon: float
                To classify all the points in the training set as +1,
                the model will set the dividing line (threshold) to
                threshold = min(x_best_feature) - epsilon
        """
        self.epsilon = epsilon
        self.best_feature = ''  # label of the best feature column
        self.threshold = 0.0  # dividing line
        self.inverse = False

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              weights: pd.Series = None, full_error: bool = False):
        n_data = len(X_train)
        # Compute errors for all possible dividing lines
        errors = []
        for feature in X_train.columns:
            x_col = X_train[feature]
            # Iterate over all data points
            err = [self.weighted_error(y_train,
                                       self._predict(x_col,
                                                     threshold=xi,
                                                     inverse=False),
                                       weights)
                   for xi in x_col]
            # Set the threshold below the minimum of current feature
            threshold_min = min(x_col) - self.epsilon
            y_pred = self._predict(x_col, threshold=threshold_min, inverse=False)
            err.append(self.weighted_error(y_train, y_pred, weights))
            # Store the errors
            errors.append(pd.Series(err, name=f"{feature}"))
            # Inverse the decision
            # Iterate over all data points
            err = [self.weighted_error(y_train,
                                       self._predict(x_col,
                                                     threshold=xi,
                                                     inverse=True),
                                       weights)
                   for xi in x_col]
            # Set the threshold below the minimum of current feature
            threshold_min = min(x_col) - self.epsilon
            y_pred = self._predict(x_col, threshold=threshold_min, inverse=True)
            err.append(self.weighted_error(y_train, y_pred, weights))
            # Store the errors
            errors.append(pd.Series(err, name=f"{feature}-inverse"))

        errors = pd.DataFrame(errors).T
        errors_arr = errors.to_numpy()
        # Find the minimizer of the errors
        best_data, best_feature = np.unravel_index(np.argmin(errors_arr, axis=None),
                                                   errors_arr.shape)
        err_min = errors_arr[best_data, best_feature]
        # Store parameters
        self.inverse = bool(best_feature % 2)  # odd columns
        self.best_feature = X_train.columns[best_feature // 2]
        if best_data == n_data:  # last error corresponds to the minimum threshold
            self.threshold = min(X_train[self.best_feature]) - self.epsilon
        else:
            self.threshold = X_train[self.best_feature][best_data]
        # Return the errors
        if full_error:
            return errors, err_min
        else:
            return err_min

    def eval_model(self, X_test: pd.DataFrame, y_test: pd.Series,
                   weights: pd.Series = None) -> Tuple[pd.Series, float]:
        y_pred = self.predict(X_test)
        error = self.weighted_error(y_test, y_pred, weights)
        return y_pred, error

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        return self._predict(X_test[self.best_feature],
                             self.threshold, self.inverse)

    @staticmethod
    def _predict(x: pd.Series, threshold: float, inverse: bool):
        if inverse:
            y_pred = 2 * (x <= threshold) - 1
        else:
            y_pred = 2 * (x > threshold) - 1
        y_pred.name = 'y_pred'
        return y_pred

    @staticmethod
    def weighted_error(y_true: pd.Series, y_pred: pd.Series,
                       weights: pd.Series = None) -> float:
        if weights is None:
            return np.average(y_true != y_pred)
        else:
            return weights.dot(y_pred != y_true)
