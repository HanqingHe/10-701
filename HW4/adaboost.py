from typing import Tuple

import numpy as np
import pandas as pd

from HW4.decisionstump import DecisionStump


class Adaboost:
    def __init__(self):
        self.T = 0
        self.h = []
        self.alpha = pd.Series([])
        self.w = pd.DataFrame([])

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              n_iter: int = 10):
        # Initialize parameters
        N, D = X_train.shape
        self.T = n_iter
        self.h = []
        self.alpha = []
        self.w = []
        w_t = pd.Series(np.full(N, 1/N), index=y_train.index, name=f"iter 0")
        # Boosting
        for t in range(self.T):
            h_t = DecisionStump()
            # Compute the weighted training error of h_t
            err_t = h_t.train(X_train, y_train, w_t)
            # Compute the importance of h_t
            alpha_t = 0.5 * np.log((1 - err_t) / err_t)
            # Update the weights
            h_t_pred = h_t.predict(X_train)
            w_t = w_t * np.exp(-alpha_t * y_train * h_t_pred)
            w_t = w_t / w_t.sum()
            w_t = pd.Series(w_t, index=y_train.index, name=f"iter {t+1}")
            # Store parameters
            self.h.append(h_t)
            self.alpha.append(alpha_t)
            self.w.append(w_t)
        self.alpha = pd.Series(self.alpha, name='importance')
        self.w = pd.DataFrame(self.w).T

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        # h_pred: shape=(T, N), predictions by all the weak classifiers
        h_pred = np.array([h_t.predict(X_test).to_numpy() for h_t in self.h])
        # alpha: shape=(T,), importance of each weak classifier
        alpha = np.array(self.alpha)
        y_pred = np.sign(np.einsum('ti,t->i', h_pred, alpha))
        y_pred = pd.Series(y_pred, index=X_test.index, name='y_pred')
        return y_pred

    def eval_model(self, X_test: pd.DataFrame, y_test: pd.Series, full: bool = False):
        if not full:
            y_pred = self.predict(X_test)
            acc = self.acc(y_pred, y_test)
            return y_pred, acc
        else:
            y_preds = []
            accs = []
            for t in range(self.T):
                alpha, h = self.alpha[:t+1], self.h[:t+1]
                y_pred = self._predict(X_test, alpha, h)
                y_pred.name = f"iter {t+1}"
                y_preds.append(y_pred)
                accs.append(self.acc(y_pred, y_test))
            y_preds = pd.DataFrame(y_preds).T
            accs = pd.Series(accs, name='accuracy')
            return y_preds, accs

    @staticmethod
    def acc(y_pred: pd.Series, y_true: pd.Series) -> float:
        return np.average(y_pred.to_numpy() == y_true.to_numpy())

    @staticmethod
    def _predict(X_test: pd.DataFrame, alpha: pd.Series, h: list):
        # h_pred: shape=(T, N), predictions by all the weak classifiers
        h_pred = np.array([h_t.predict(X_test).to_numpy() for h_t in h])
        # alpha: shape=(T,), importance of each weak classifier
        alpha = np.array(alpha)
        y_pred = np.sign(np.einsum('ti,t->i', h_pred, alpha))
        y_pred = pd.Series(y_pred, index=X_test.index, name='y_pred')
        return y_pred

