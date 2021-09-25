import numpy as np
import pandas as pd


class NB:
    def __init__(self):
        self.target = ""  # name of the label
        self.columns = pd.Index([])  # name of the features
        self.num_cols = pd.Index([])  # name of numerical features
        self.cat_cols = pd.Index([])  # name of categorical features
        self.py = {}  # P(y)
        self.px = {}  # P(xi|y)

    def train(self, X: pd.DataFrame, y: pd.Series):
        # Sanity check
        assert all(X.index == y.index), "Indices mismatch"
        # Drop rows with missing data
        Xy = pd.concat([X, y], axis=1).dropna(axis=0, how='any')
        _X, _y = Xy[X.columns], Xy[y.name]
        # Initialization
        self.target = _y.name
        self.columns = _X.columns
        self.num_cols = _X.select_dtypes(include='number').columns
        self.cat_cols = _X.select_dtypes(exclude='number').columns
        self.cat_cols = self.columns.drop(self.num_cols)
        # Estimate log P(y)
        y_counts = _y.value_counts()
        y_total = y_counts.sum()
        self.py = {y_val: y_count / y_total for y_val, y_count in y_counts.iteritems()}
        # Estimate log P(xi|y)
        for y_val, py in self.py.items():
            self.px[y_val] = {}
            X_given_y = _X[_y == y_val]
            # Split X_given_y into numerical and categorical parts
            X_num_given_y = X_given_y[self.num_cols]
            X_cat_given_y = X_given_y[self.cat_cols]
            # Numerical: mean and standard deviation
            self.px[y_val]['numerical'] = X_num_given_y.describe().loc[['mean', 'std'], :]
            # Categorical: frequency
            self.px[y_val]['categorical'] = {feature: xi.value_counts(normalize=True)
                                             for feature, xi in X_cat_given_y.iteritems()}

    def predict(self, X: pd.DataFrame, return_LL: bool = False):
        r"""Predict the labels of all the instances in a feature matrix

        Args:
            X: pd.DataFrame
            return_LL: bool
                If set to True, return the log-posterior

        Returns:
            pred (return_LL=False)
            pred, LL (return_LL=True)
        """
        pred = []
        LL = []
        for index, x in X.iterrows():
            # Compute log-likelihood
            ll = self.LL_numerical(x) + self.LL_categorical(x)
            # Find the most likely label
            ll.sort_values(ascending=False, inplace=True)
            LL.append(ll)
            if np.inf in ll.values:  # xi contains values not included by the training set
                # Break ties by comparing P(y)
                pred.append(pd.Series(self.py).sort_values(ascending=False).index[0])
            else:
                pred.append(ll.index[0])
        # Clean up LL and pred
        LL = pd.concat(LL, axis=1).T
        LL.index = X.index
        pred = pd.Series(pred, index=X.index, name=self.target)
        if return_LL:
            return pred, LL
        else:
            return pred

    def LL_numerical(self, x: pd.Series) -> pd.Series:
        r"""Log-likelihood of all numerical features of a given instance

        Args:
            x: pd.Series

        Returns:
            pd.Series
        """
        _num_cols = self.num_cols.drop(x.index[x.isna()], errors='ignore')
        _x = x[_num_cols].to_numpy()
        _ll = {}
        for (y_val, px), py in zip(self.px.items(), self.py.values()):
            _mu = px['numerical'].loc['mean', _num_cols].to_numpy()
            _sigma = px['numerical'].loc['std', _num_cols].to_numpy()
            _ll[y_val] = np.sum(self.log_gaussian(_x, _mu, _sigma))
        return pd.Series(_ll)

    def LL_categorical(self, x: pd.Series) -> pd.Series:
        r"""Log-posterior of all categorical features of a given instance

        Args:
            x: pd.Series

        Returns:
            pd.Series
        """
        _cat_cols = self.cat_cols.drop(x.index[x.isna()], errors='ignore')
        _x = x[_cat_cols]
        _ll = {}
        for (y_val, px), py in zip(self.px.items(), self.py.values()):
            px_given_y = [px['categorical'][feature].get(xi, 0) for feature, xi in _x.iteritems()]
            _ll[y_val] = np.sum(np.log(px_given_y)) + np.log(py)
        return pd.Series(_ll)

    @staticmethod
    def log_gaussian(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        r"""Log-probability from a Gaussian distribution

        Args:
            x: np.ndarray
            mean: np.ndarray
            std: np.ndarray

        Returns:
            res
        """
        epsilon = 1e-9
        mu = mean
        s2 = np.square(std) + epsilon
        c0 = -1/2 * np.log(2 * np.pi * s2)
        A = -np.square(x - mu) / (2 * s2)
        res = c0 + A
        return res
