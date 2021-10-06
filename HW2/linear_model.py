import numpy as np
import pandas as pd


class LinearRegression:
    def __init__(self, reg: str = None, pen: float = None):
        # Sanity check
        if reg not in (None, 'l1', 'l2'):
            raise ValueError('Regularization not supported')
        if reg in ('l1', 'l2'):
            try:
                assert pen > 0
            except (AssertionError, TypeError):
                raise ValueError('Penalty must be positive when regularization is specified')
        self.reg = reg  # type of regularization, can be None, 'l1' or 'l2'
        self.pen = pen  # penalty (lambda), must be positive for L1 or L2 regularization
        self.features = pd.Index([])  # name of the features
        self.target = pd.Index([])  # name of the target
        self.w = pd.Series([])  # weights
        self.b = 0.0  # bias
        self.lr = 0.0  # learning rate, must be positive
        self.n_epoch = 0  # number of epochs, must be a positive integer

    def train(self, X: pd.DataFrame, y: pd.Series,
              lr: float = 0.01, n_epoch: int = 50) -> pd.Series:
        r"""Train the model to data

        Args:
            X: pd.DataFrame
                Feature matrix
            y: pd.Series
                Target vector
            lr: float
                Learning rate, must be positive
            n_epoch: int
                Number of epochs, must be a positive integer

        Returns:
            losses: pd.Series
                Training loss per data point
        """
        # Sanity check
        try:
            assert lr > 0
        except (AssertionError, TypeError):
            raise ValueError('Learning rate must be positive')
        try:
            assert isinstance(n_epoch, int)
            assert n_epoch > 0
        except AssertionError:
            raise ValueError('Number of epochs must be a positive integer')
        # Initialize weights and bias
        self.features = X.columns
        self.target = y.name
        self.w = pd.Series(np.zeros(len(self.features)), name='w', index=self.features)
        self.b = 0.0
        self.lr = lr
        self.n_epoch = n_epoch
        # Train model with SGD
        losses = []
        for i_epoch in range(n_epoch):
            for (i_data, xi), (_, yi) in zip(X.iterrows(), y.iteritems()):
                _X = xi.to_frame().T
                _y = pd.Series(yi, index=[i_data], name=self.target)
                # Compute and store loss
                _loss = self.loss(_X, _y)
                losses.append(_loss)
                # Update parameters
                _grad = self.gradient(_X, _loss)
                # Flatten dw and db since xi is 1-D (single data point)
                dw = _grad['dw'].iloc[0]
                db = _grad['db'].iloc[0]
                # Update parameters
                self.w = self.w - self.lr * dw
                self.b = self.b - self.lr * db
        # Clean up
        losses = pd.concat(losses, ignore_index=True)
        return losses

    def predict(self, X: pd.DataFrame) -> pd.Series:
        r"""Predict the target given features

        Args:
            X: pd.DataFrame
                Feature matrix

        Returns:
            y_pred: pd.Series
                Predictions of the target
        """
        _X = X.to_numpy()
        y_pred = X.dot(self.w) + self.b  # broadcast w and b
        y_pred = pd.Series(y_pred, index=X.index, name=self.target)
        return y_pred

    def loss(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        r"""Difference between predictions and ground truth

        Args:
            X: pd.DataFrame
                Feature matrix of the test set
            y: pd.Series
                Target vector of the test set
        Returns:
            losses: pd.Series
                Difference between model predictions and ground truth
        """
        y_pred = self.predict(X)
        losses = y_pred - y
        losses.name = 'loss'
        return losses

    def gradient(self, X: pd.DataFrame, loss: pd.Series) -> dict:
        r"""Gradients of the loss function wrt each parameter

        Args:
            X: pd.DataFrame
                Feature matrix
            loss: pd.Series
                Loss vector

        Returns:
            grad: dict
                Gradients of the loss function wrt each parameter
        """
        # dJ/dw
        _X = X.to_numpy()
        _loss = loss.to_numpy()
        # dJ/dw = 2 (y_pred - y) x
        dw = 2 * _loss[:, np.newaxis] * _X
        if self.reg is None:
            pass
        elif self.reg == 'l1':
            # dJ/dw = 2 (y_pred - y) x + lambda * sgn w
            dw += self.pen * np.sign(self.w)
        elif self.reg == 'l2':
            # dJ/dw = 2 (y_pred - y) x + 2 * lambda * w
            dw += 2 * self.pen * self.w
        else:
            raise NotImplementedError
        # dJ/db
        db = 2 * loss
        # Clean up
        dw = pd.DataFrame(dw, index=X.index, columns=X.columns)
        db = pd.Series(db, index=X.index, name='db')
        # Collect both derivatives in a dictionary
        grad = {'dw': dw,
                'db': db}
        return grad
